"""POST /api/v1/search -- the retrieval endpoint.

`mode="pipeline"` is the real contract the generation layer will call:
tenant-scoped hybrid retrieval + RRF -> rerank -> token-budget context,
returning the chunks with full citation metadata and per-phase timings.
`mode="vector" | "fts" | "hybrid"` expose the individual stages for
inspection and for the demo (showing raw hits before generation).
"""

import time
from dataclasses import asdict
from typing import Literal
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import DbSession, EmbedderDep, RerankerDep
from app.services.retrieval import (
    fts_search,
    hybrid_search,
    retrieve_context,
    vector_search,
)

router = APIRouter(prefix="/api/v1/search", tags=["search"])


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    tenant_id: UUID
    mode: Literal["vector", "fts", "hybrid", "pipeline"] = "vector"
    strategy: Literal["fixed_500", "structural"] | None = None
    k: int | None = Field(default=None, ge=1, le=100)  # hits to return / rerank_top_k
    ef_search: int | None = Field(default=None, ge=1, le=1000)  # vector / hybrid only


class SearchHit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    page_from: int
    page_to: int
    section_path: str | None
    heading: str | None
    score: float


class PipelineTimingOut(BaseModel):
    hybrid_ms: float
    rerank_ms: float
    select_ms: float
    total_ms: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    took_ms: float
    timing: PipelineTimingOut | None = None  # populated for mode="pipeline"


@router.post("", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    session: DbSession,
    embedder: EmbedderDep,
    reranker: RerankerDep,
) -> SearchResponse:
    started = time.perf_counter()
    timing: PipelineTimingOut | None = None
    if payload.mode == "fts":
        chunks = await fts_search(
            session,
            tenant_id=payload.tenant_id,
            question=payload.question,
            strategy=payload.strategy,
            k=payload.k,
        )
    elif payload.mode == "hybrid":
        chunks = await hybrid_search(
            session,
            embedder,
            tenant_id=payload.tenant_id,
            question=payload.question,
            strategy=payload.strategy,
            top_k=payload.k,
            ef_search=payload.ef_search,
        )
    elif payload.mode == "pipeline":
        result = await retrieve_context(
            session,
            embedder,
            reranker,
            tenant_id=payload.tenant_id,
            question=payload.question,
            strategy=payload.strategy,
            rerank_top_k=payload.k,
        )
        chunks = result.context
        timing = PipelineTimingOut(**asdict(result.timing))
    else:
        chunks = await vector_search(
            session,
            embedder,
            tenant_id=payload.tenant_id,
            question=payload.question,
            strategy=payload.strategy,
            k=payload.k,
            ef_search=payload.ef_search,
        )
    took_ms = (time.perf_counter() - started) * 1000
    return SearchResponse(
        hits=[SearchHit(**asdict(chunk)) for chunk in chunks],
        took_ms=round(took_ms, 1),
        timing=timing,
    )
