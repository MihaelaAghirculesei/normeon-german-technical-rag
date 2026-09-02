"""Retrieval pipeline.

Three branches -- vector k-NN, German full-text, and a trigram code
fallback -- exposed as standalone callables, fused by `hybrid_search`
with Reciprocal Rank Fusion, and composed by `retrieve_context` into the
full instrumented flow: tenant-scoped hybrid retrieval + RRF -> rerank ->
token-budget context selection.
"""

import asyncio
import time
from dataclasses import replace
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embedding.base import EmbeddingAdapter
from app.adapters.reranker.base import Reranker
from app.core.config import settings
from app.db.models import EMBEDDING_DIM
from app.db.queries import load_sql
from app.domain.fusion import rrf
from app.domain.models import PipelineTiming, RetrievalResult, RetrievedChunk
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


async def _fts_branch(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    question: str,
    strategy: str,
    k: int,
) -> list[RetrievedChunk]:
    """The `websearch_to_tsquery('german')` / `ts_rank_cd` branch on its
    own. The question goes through the same `normalize_de` as the indexed
    text, so a code or norm reference written differently still lands on
    the token ingestion stored.
    """
    result = await session.execute(
        _FTS_SEARCH_SQL,
        {
            "q": normalize_de(question),
            "tenant_id": tenant_id,
            "strategy": strategy,
            "k": k,
        },
    )
    return [_row_to_chunk(row) for row in result]


async def _trgm_branch(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    question: str,
    strategy: str,
    k: int,
) -> list[RetrievedChunk]:
    """Trigram `word_similarity` hits for the code the question names, or
    `[]` when it names none. Extra recall for a lightly misspelled or
    oddly spaced code that the German FTS tokenizer fumbles.
    """
    code = extract_code(question)
    if code is None:
        return []
    await session.execute(
        text(
            "SET LOCAL pg_trgm.word_similarity_threshold = "
            f"{float(settings.trgm_code_threshold)}"
        )
    )
    result = await session.execute(
        _TRGM_SEARCH_SQL,
        {"code": code, "tenant_id": tenant_id, "strategy": strategy, "k": k},
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

    When the question looks like it names a code (`extract_code`), the
    trigram branch runs as well and its chunks are appended (deduped)
    after the full-text hits. This is a plain concatenation, not a
    score-aware merge -- `hybrid_search` is where the two become
    score-comparable via RRF.
    """
    strategy = strategy or settings.retrieval_strategy
    k = k if k is not None else settings.retrieval_top_k

    hits = await _fts_branch(
        session, tenant_id=tenant_id, question=question, strategy=strategy, k=k
    )
    trgm_hits = await _trgm_branch(
        session, tenant_id=tenant_id, question=question, strategy=strategy, k=k
    )
    if trgm_hits:
        seen = {hit.chunk_id for hit in hits}
        hits.extend(hit for hit in trgm_hits if hit.chunk_id not in seen)

    return hits[:k]


async def hybrid_search(
    session: AsyncSession,
    embedder: EmbeddingAdapter,
    *,
    tenant_id: UUID,
    question: str,
    strategy: str | None = None,
    candidate_k: int | None = None,
    top_k: int | None = None,
    ef_search: int | None = None,
    rrf_k: int | None = None,
    weights: tuple[float, float, float] | None = None,
) -> list[RetrievedChunk]:
    """Vector, full-text and trigram retrieval fused with Reciprocal Rank
    Fusion.

    Pulls `candidate_k` hits from each branch, fuses them on `chunk_id`,
    and returns the top `top_k` as `RetrievedChunk` whose `score` is the
    fused RRF score. Unlike `fts_search`, the trigram branch is a third
    ranking in the fusion here, not concatenated onto the full-text list;
    RRF is what makes the three branches' scores comparable.

    `rrf_k` and the per-branch `weights` (vector, full-text, trigram)
    default to the configured values and are experiment-matrix variables
    in Week 4. The branches run sequentially: they share one AsyncSession,
    which is not safe for concurrent statements, and each branch's
    `SET LOCAL` knobs stay scoped to their own statement anyway.
    """
    strategy = strategy or settings.retrieval_strategy
    candidate_k = candidate_k if candidate_k is not None else settings.hybrid_candidate_k
    top_k = top_k if top_k is not None else settings.hybrid_top_k
    rrf_k = rrf_k if rrf_k is not None else settings.rrf_k
    if weights is None:
        weights = (
            settings.rrf_weight_vector,
            settings.rrf_weight_fts,
            settings.rrf_weight_trgm,
        )

    vec_hits = await vector_search(
        session,
        embedder,
        tenant_id=tenant_id,
        question=question,
        strategy=strategy,
        k=candidate_k,
        ef_search=ef_search,
    )
    fts_hits = await _fts_branch(
        session, tenant_id=tenant_id, question=question, strategy=strategy, k=candidate_k
    )
    trgm_hits = await _trgm_branch(
        session, tenant_id=tenant_id, question=question, strategy=strategy, k=candidate_k
    )

    by_id: dict[str, RetrievedChunk] = {}
    for branch in (vec_hits, fts_hits, trgm_hits):
        for hit in branch:
            by_id.setdefault(str(hit.chunk_id), hit)

    fused = rrf(
        [
            [str(hit.chunk_id) for hit in vec_hits],
            [str(hit.chunk_id) for hit in fts_hits],
            [str(hit.chunk_id) for hit in trgm_hits],
        ],
        k=rrf_k,
        weights=list(weights),
    )
    return [replace(by_id[chunk_id], score=score) for chunk_id, score in fused[:top_k]]


def _select_context(
    chunks: list[RetrievedChunk], token_budget: int
) -> list[RetrievedChunk]:
    """Fill a token budget with the highest-ranked chunks, in order.

    A `section_path` already represented is skipped, so the context is not
    three near-duplicate slices of one section. Chunks are taken whole;
    selection stops once the next chunk would push the running total over
    the budget. At least the top chunk is always returned, even if it
    alone exceeds the budget.

    Token count is the whitespace-word approximation used throughout
    (chunking's `_estimate_tokens`); the budget is a soft target, not a
    hard model-context limit.
    """
    selected: list[RetrievedChunk] = []
    seen_sections: set[str] = set()
    used = 0
    for chunk in chunks:
        if chunk.section_path is not None and chunk.section_path in seen_sections:
            continue
        cost = len(chunk.content.split())
        if selected and used + cost > token_budget:
            break
        selected.append(chunk)
        used += cost
        if chunk.section_path is not None:
            seen_sections.add(chunk.section_path)
    return selected


async def retrieve_context(
    session: AsyncSession,
    embedder: EmbeddingAdapter,
    reranker: Reranker,
    *,
    tenant_id: UUID,
    question: str,
    strategy: str | None = None,
    candidate_k: int | None = None,
    rerank_top_k: int | None = None,
    token_budget: int | None = None,
) -> RetrievalResult:
    """The full retrieval pipeline as one instrumented call:

        tenant filter + hybrid retrieval + RRF   (`hybrid_search`)
          -> cross-encoder rerank                (off-thread, CPU-bound)
          -> token-budget context selection      (`_select_context`)

    Per-phase wall-clock timings travel back in `RetrievalResult.timing`.
    `rerank_top_k` and `token_budget` fall back to the configured
    defaults; `candidate_k` is forwarded to `hybrid_search` (its own
    default decides how many candidates the reranker sees).
    """
    rerank_top_k = rerank_top_k if rerank_top_k is not None else settings.rerank_top_k
    token_budget = (
        token_budget if token_budget is not None else settings.context_token_budget
    )

    started = time.perf_counter()
    fused = await hybrid_search(
        session,
        embedder,
        tenant_id=tenant_id,
        question=question,
        strategy=strategy,
        candidate_k=candidate_k,
    )
    after_hybrid = time.perf_counter()
    reranked = await asyncio.to_thread(reranker.rerank, question, fused, rerank_top_k)
    after_rerank = time.perf_counter()
    context = _select_context(reranked, token_budget)
    after_select = time.perf_counter()

    return RetrievalResult(
        context=context,
        reranked=reranked,
        timing=PipelineTiming(
            hybrid_ms=round((after_hybrid - started) * 1000, 1),
            rerank_ms=round((after_rerank - after_hybrid) * 1000, 1),
            select_ms=round((after_select - after_rerank) * 1000, 1),
            total_ms=round((after_select - started) * 1000, 1),
        ),
    )
