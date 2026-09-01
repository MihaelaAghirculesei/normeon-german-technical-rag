"""Debug endpoint for raw vector retrieval. Not part of the answer flow --
it exists to inspect what the retriever returns for a question, and it's
useful enough in the demo (showing the raw hits before generation) to keep
around.
"""

import time
from dataclasses import asdict
from typing import Literal
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import DbSession, EmbedderDep
from app.services.retrieval import vector_search

router = APIRouter(prefix="/api/v1/search", tags=["search"])


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    tenant_id: UUID
    strategy: Literal["fixed_500", "structural"] | None = None
    k: int | None = Field(default=None, ge=1, le=100)
    ef_search: int | None = Field(default=None, ge=1, le=1000)


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


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    took_ms: float


@router.post("", response_model=SearchResponse)
async def search(
    payload: SearchRequest, session: DbSession, embedder: EmbedderDep
) -> SearchResponse:
    started = time.perf_counter()
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
    )
