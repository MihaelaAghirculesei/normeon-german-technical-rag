from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embedding.base import EmbeddingAdapter
from app.adapters.embedding.e5_api import ApiE5Embedder
from app.adapters.embedding.e5_local import LocalE5Embedder
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
