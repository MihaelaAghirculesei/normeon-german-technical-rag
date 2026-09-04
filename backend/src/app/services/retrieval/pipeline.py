"""The full retrieval pipeline: hybrid retrieval -> rerank -> token-budget
context selection, as one instrumented call.
"""

import asyncio
import time
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embedding.base import EmbeddingAdapter
from app.adapters.reranker.base import Reranker
from app.core.config import settings
from app.domain.models import PipelineTiming, RetrievalResult, RetrievedChunk
from app.services.retrieval.hybrid import hybrid_search


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
