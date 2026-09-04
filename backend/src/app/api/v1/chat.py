"""POST /api/v1/chat -- the non-streaming answer endpoint (plan, Giorno 11).

The question is retrieved, the surviving chunks become numbered sources,
the versioned German prompt is rendered around them, and the model is
asked for an answer that cites only `[S1]`, `[S2]`, ... . The model never
sees or produces page numbers: the response carries the `marker ->
page/section/document` mapping the backend held back, which is what makes
the citations verifiable.

Citation validation (dropping invented markers, abstaining on zero valid
citations) is Day 12; the pre-generation confidence gate is Day 13;
streaming is Day 14.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import DbSession, EmbedderDep, LlmClientDep, RerankerDep
from app.services.generation import generate_answer

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    tenant_id: UUID
    strategy: Literal["fixed_500", "structural"] | None = None


class SourceOut(BaseModel):
    marker: str
    document_id: UUID
    filename: str
    page_from: int
    page_to: int
    section_path: str | None
    heading: str | None


class RetrievalTimingOut(BaseModel):
    hybrid_ms: float
    rerank_ms: float
    select_ms: float
    total_ms: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    prompt_name: str
    prompt_sha256: str
    model: str
    retrieval_timing: RetrievalTimingOut
    generation_ms: float


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    session: DbSession,
    embedder: EmbedderDep,
    reranker: RerankerDep,
    llm: LlmClientDep,
) -> ChatResponse:
    result = await generate_answer(
        session,
        embedder,
        reranker,
        llm,
        tenant_id=payload.tenant_id,
        question=payload.question,
        strategy=payload.strategy,
    )
    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceOut(
                marker=s.marker,
                document_id=s.document_id,
                filename=s.filename,
                page_from=s.page_from,
                page_to=s.page_to,
                section_path=s.section_path,
                heading=s.heading,
            )
            for s in result.sources
        ],
        prompt_name=result.prompt_name,
        prompt_sha256=result.prompt_sha256,
        model=result.model,
        retrieval_timing=RetrievalTimingOut(**asdict(result.retrieval_timing)),
        generation_ms=result.generation_ms,
    )
