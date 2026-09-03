"""German full-text retrieval and the trigram code fallback."""

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.domain.models import RetrievedChunk
from app.domain.normalization import extract_code, normalize_de
from app.services.retrieval._sql import FTS_SEARCH_SQL, TRGM_SEARCH_SQL, row_to_chunk


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
        FTS_SEARCH_SQL,
        {
            "q": normalize_de(question),
            "tenant_id": tenant_id,
            "strategy": strategy,
            "k": k,
        },
    )
    return [row_to_chunk(row) for row in result]


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
        TRGM_SEARCH_SQL,
        {"code": code, "tenant_id": tenant_id, "strategy": strategy, "k": k},
    )
    return [row_to_chunk(row) for row in result]


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
