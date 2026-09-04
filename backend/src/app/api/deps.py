from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embedding.base import EmbeddingAdapter
from app.adapters.embedding.e5_api import ApiE5Embedder
from app.adapters.embedding.e5_local import LocalE5Embedder
from app.adapters.llm.base import LlmClient
from app.adapters.llm.fake import FakeLlmClient
from app.adapters.llm.openai_compatible import OpenAICompatibleClient
from app.adapters.reranker.base import Reranker
from app.adapters.reranker.cross_encoder import CrossEncoderReranker
from app.adapters.reranker.noop import NoopReranker
from app.core.config import settings
from app.db.session import get_session

DbSession = Annotated[AsyncSession, Depends(get_session)]


@lru_cache
def get_embedder() -> EmbeddingAdapter:
    """Cached so the (heavy, lazily-loaded) local model is constructed once
    per process rather than once per request."""
    if settings.embedding_provider == "e5_local":
        return LocalE5Embedder(settings.embedding_model, settings.embedding_dim)
    if settings.embedding_api_base_url is None:
        raise RuntimeError("embedding_api_base_url must be set when embedding_provider=e5_api")
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
    return CrossEncoderReranker(settings.reranker_model)


RerankerDep = Annotated[Reranker, Depends(get_reranker)]


@lru_cache
def get_llm_client() -> LlmClient:
    """Cached for the process. `fake` needs nothing (deterministic, no
    network); `openai_compatible` needs a base URL and a model -- any
    server speaking POST /chat/completions."""
    if settings.llm_provider == "fake":
        return FakeLlmClient()
    if not settings.llm_api_base_url or not settings.llm_model:
        raise RuntimeError(
            "llm_api_base_url and llm_model must be set when llm_provider=openai_compatible"
        )
    return OpenAICompatibleClient(
        base_url=settings.llm_api_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout_s,
    )


LlmClientDep = Annotated[LlmClient, Depends(get_llm_client)]
