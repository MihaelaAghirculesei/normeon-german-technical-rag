"""Vector k-NN retrieval over the HNSW index."""

import asyncio
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embedding.base import EmbeddingAdapter
from app.core.config import settings
from app.domain.models import RetrievedChunk
from app.services.retrieval._sql import VECTOR_SEARCH_SQL, row_to_chunk


async def vector_search(
    session: AsyncSession,
    embedder: EmbeddingAdapter,
    *,
    tenant_id: UUID,
    question: str,
    strategy: str | None = None,
    k: int | None = None,
    ef_search: int | None = None,
) -> list[RetrievedChunk]:
    """Embed the question and return the k nearest chunks for this tenant,
    ranked by cosine similarity. `strategy` / `k` / `ef_search` fall back to
    the configured defaults.
    """
    query_vector = await asyncio.to_thread(embedder.embed_query, question)
    return await vector_search_by_vector(
        session,
        tenant_id=tenant_id,
        query_vector=query_vector,
        model=embedder.name,
        strategy=strategy,
        k=k,
        ef_search=ef_search,
    )


async def vector_search_by_vector(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    query_vector: list[float],
    model: str,
    strategy: str | None = None,
    k: int | None = None,
    ef_search: int | None = None,
) -> list[RetrievedChunk]:
    """The DB half of `vector_search`, split out so a caller that has
    already embedded the query (a script measuring retrieval latency on its
    own, or a future multi-strategy fusion step) doesn't embed it again.

    Two session-local pgvector knobs are set first:
    - `hnsw.ef_search` -- recall/latency trade-off for the HNSW scan.
    - `hnsw.iterative_scan = relaxed_order` -- without it the tenant /
      strategy filters are applied *after* the fixed-size index scan, so a
      selective filter can yield fewer than `k` rows; relaxed_order lets
      pgvector keep scanning until it has enough. `relaxed` (not `strict`)
      because the query re-sorts by score anyway.

    `SET LOCAL` scopes both to the current transaction, which the session
    autobegins on this first statement.
    """
    strategy = strategy or settings.retrieval_strategy
    k = k if k is not None else settings.retrieval_top_k
    ef_search = ef_search if ef_search is not None else settings.hnsw_ef_search

    await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))

    result = await session.execute(
        VECTOR_SEARCH_SQL,
        {
            "qvec": query_vector,
            "tenant_id": tenant_id,
            "strategy": strategy,
            "model": model,
            "k": k,
        },
    )
    return [row_to_chunk(row) for row in result]
