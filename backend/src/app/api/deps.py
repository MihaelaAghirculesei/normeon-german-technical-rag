from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embedding.base import EmbeddingAdapter
from app.adapters.embedding.e5_api import ApiE5Embedder
from app.adapters.embedding.e5_local import LocalE5Embedder
from app.adapters.embedding.gemini import GeminiEmbedder
from app.adapters.llm.base import LlmClient
from app.adapters.llm.fake import FakeLlmClient
from app.adapters.llm.openai_compatible import OpenAICompatibleClient
from app.adapters.reranker.base import Reranker
from app.adapters.reranker.cross_encoder import CrossEncoderReranker
from app.adapters.reranker.noop import NoopReranker
from app.core.config import settings
from app.core.errors import (
    EmbeddingMisconfiguredError,
    ForbiddenError,
    LlmMisconfiguredError,
    NotAuthenticatedError,
)
from app.core.security import AuthContext, decode_access_token
from app.db.session import get_session, tenant_session

# Unscoped: login and health only. Anything touching tenant data uses
# TenantSession, which also requires a valid token.
DbSession = Annotated[AsyncSession, Depends(get_session)]

_bearer = HTTPBearer(auto_error=False)


def get_auth(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> AuthContext:
    if credentials is None:
        raise NotAuthenticatedError("missing bearer token")
    return decode_access_token(credentials.credentials)


AuthDep = Annotated[AuthContext, Depends(get_auth)]


async def get_tenant_session(auth: AuthDep) -> AsyncIterator[AsyncSession]:
    """The caller's tenant, from the token, confines every transaction of
    this session (row-level security, migration 0004)."""
    async with tenant_session(auth.tenant_id) as session:
        yield session


TenantSession = Annotated[AsyncSession, Depends(get_tenant_session)]


def require_admin(auth: AuthDep) -> AuthContext:
    if auth.role != "admin":
        raise ForbiddenError("this action needs the admin role")
    return auth


AdminDep = Annotated[AuthContext, Depends(require_admin)]


@lru_cache
def get_embedder() -> EmbeddingAdapter:
    """Cached so the (heavy, lazily-loaded) local model is constructed once
    per process rather than once per request."""
    if settings.embedding_provider == "e5_local":
        return LocalE5Embedder(settings.embedding_model, settings.embedding_dim)
    if settings.embedding_provider == "gemini":
        if not settings.embedding_api_key:
            raise EmbeddingMisconfiguredError(
                "embedding_api_key must be set when embedding_provider=gemini"
            )
        return GeminiEmbedder(
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
    if settings.embedding_api_base_url is None:
        raise EmbeddingMisconfiguredError(
            "embedding_api_base_url must be set when embedding_provider=e5_api"
        )
    return ApiE5Embedder(
        base_url=settings.embedding_api_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )


EmbedderDep = Annotated[EmbeddingAdapter, Depends(get_embedder)]


@lru_cache
def get_reranker() -> Reranker:
    """Cached so the cross-encoder is built once per process, not per
    request (the spec's "singleton for the app's lifetime"). The model
    itself still loads lazily inside the adapter on first use."""
    if settings.reranker_provider == "noop":
        return NoopReranker()
    return CrossEncoderReranker(settings.reranker_model, settings.reranker_max_length)


RerankerDep = Annotated[Reranker, Depends(get_reranker)]


@lru_cache
def get_llm_client() -> LlmClient:
    """Cached for the process. `fake` needs nothing (deterministic, no
    network); `openai_compatible` needs a base URL and a model -- any
    server speaking POST /chat/completions."""
    return build_llm_client()


def build_llm_client(model: str | None = None) -> LlmClient:
    """A client for the configured provider, on `model` instead of the
    configured one when given -- the evaluation compares models while its
    judge stays on the configured one."""
    if settings.llm_provider == "fake":
        return FakeLlmClient()
    model = model or settings.llm_model
    if not settings.llm_api_base_url or not model:
        raise LlmMisconfiguredError(
            "llm_api_base_url and llm_model must be set when llm_provider=openai_compatible"
        )
    return OpenAICompatibleClient(
        base_url=settings.llm_api_base_url,
        api_key=settings.llm_api_key,
        model=model,
        timeout=settings.llm_timeout_s,
        call_deadline=settings.llm_call_deadline_s,
        first_token_timeout=settings.llm_stream_first_token_s,
        stream_idle_timeout=settings.llm_stream_idle_s,
        max_retries=settings.llm_max_retries,
    )


LlmClientDep = Annotated[LlmClient, Depends(get_llm_client)]
