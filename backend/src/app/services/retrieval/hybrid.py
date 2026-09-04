"""Reciprocal Rank Fusion of the vector, full-text and trigram branches."""

from dataclasses import replace
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embedding.base import EmbeddingAdapter
from app.core.config import settings
from app.domain.fusion import rrf
from app.domain.models import RetrievedChunk
from app.services.retrieval.lexical import _fts_branch, _trgm_branch
from app.services.retrieval.vector import vector_search


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
