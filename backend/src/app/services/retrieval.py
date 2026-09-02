"""Retrieval pipeline. Two stages so far -- vector k-NN and German
full-text -- kept as separate callables; RRF fusion (Day 8) and reranking
(Day 9) will compose them into one instrumented function.
"""

import asyncio
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embedding.base import EmbeddingAdapter
from app.core.config import settings
from app.db.models import EMBEDDING_DIM
from app.db.queries import load_sql
from app.domain.models import RetrievedChunk
from app.domain.normalization import extract_code, normalize_de

_VECTOR_SEARCH_SQL = text(load_sql("vector_search")).bindparams(
    bindparam("qvec", type_=Vector(EMBEDDING_DIM)),
)
_FTS_SEARCH_SQL = text(load_sql("fts_search"))
_TRGM_SEARCH_SQL = text(load_sql("trgm_search"))


def _row_to_chunk(row: Any) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        filename=row.filename,
        content=row.content,
        page_from=row.page_from,
        page_to=row.page_to,
        section_path=row.section_path,
        heading=row.heading,
        score=float(row.score),
    )


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
        _VECTOR_SEARCH_SQL,
        {
            "qvec": query_vector,
            "tenant_id": tenant_id,
            "strategy": strategy,
            "model": model,
            "k": k,
        },
    )
    return [_row_to_chunk(row) for row in result]


async def fts_search(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    question: str,
    strategy: str | None = None,
    k: int | None = None,
) -> list[RetrievedChunk]:
    """German full-text search for this tenant, ranked by `ts_rank_cd`.

    The question goes through the same `normalize_de` as the indexed text,
    so a requirement code or norm reference written differently in the
    query still lands on the token that ingestion stored.

    When the question looks like it names a code (`extract_code`), a
    trigram `word_similarity` branch runs as well and its chunks are
    appended (deduped) after the full-text hits -- extra recall for a
    lightly misspelled or oddly spaced code. This is a plain concatenation,
    not a score-aware merge; true cross-branch fusion is Day 8's RRF.
    """
    strategy = strategy or settings.retrieval_strategy
    k = k if k is not None else settings.retrieval_top_k

    params = {
        "q": normalize_de(question),
        "tenant_id": tenant_id,
        "strategy": strategy,
        "k": k,
    }
    result = await session.execute(_FTS_SEARCH_SQL, params)
    hits = [_row_to_chunk(row) for row in result]

    code = extract_code(question)
    if code is not None:
        await session.execute(
            text(
                "SET LOCAL pg_trgm.word_similarity_threshold = "
                f"{float(settings.trgm_code_threshold)}"
            )
        )
        trgm_result = await session.execute(
            _TRGM_SEARCH_SQL,
            {"code": code, "tenant_id": tenant_id, "strategy": strategy, "k": k},
        )
        seen = {hit.chunk_id for hit in hits}
        hits.extend(_row_to_chunk(row) for row in trgm_result if row.chunk_id not in seen)

    return hits[:k]
