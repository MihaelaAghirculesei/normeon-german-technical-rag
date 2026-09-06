"""POST /api/v1/chat (Giorno 11) and POST /api/v1/chat/stream (Giorno 14).

The question is retrieved, the surviving chunks become numbered sources,
the versioned German prompt is rendered around them, and the model is
asked for an answer that cites only `[S1]`, `[S2]`, ... . The model never
sees or produces page numbers: the response carries the `marker ->
page/section/document` mapping the backend held back, which is what makes
the citations verifiable.

`sources` lists everything the model was offered (Giorno 11); `citations`
(Giorno 12) is the validated subset it actually cited -- an invented
marker never reaches this list, and `domain.citations.extract_and_validate`
has already turned a claim with zero valid citations into an abstention.
The pre-generation confidence gate is Day 13.

`/stream` is the same pipeline as SSE: `stage` events for retrieving /
reranking / generating, `sources` BEFORE any `token` (so a client can
show where the answer will come from while it is still being written),
then `done` (or `error` in its place). See `services.generation.
generate_answer_stream`'s docstring for what "done" means when the
validated citations disagree with what was already streamed.

`cost_usd` (Giorno 15) is the priced token cost of this answer, or
`null` when the resolved model isn't in `backend/pricing.yaml` (the
`fake` provider, or a self-hosted model with no listed price) -- see
`services.pricing`. Every answer also writes one `query_logs` row.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any, Literal
from uuid import UUID

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import DbSession, EmbedderDep, LlmClientDep, RerankerDep
from app.core.streaming import with_heartbeat
from app.domain.citations import Citation
from app.domain.context import Source
from app.services.generation import StreamEvent, generate_answer, generate_answer_stream

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
_log = structlog.get_logger(__name__)


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


class CitationOut(BaseModel):
    marker: str
    chunk_id: UUID
    document_id: UUID
    filename: str
    page_from: int
    page_to: int
    section_path: str | None
    snippet: str


class RetrievalTimingOut(BaseModel):
    hybrid_ms: float
    rerank_ms: float
    select_ms: float
    total_ms: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    citations: list[CitationOut]
    prompt_name: str
    prompt_sha256: str
    model: str
    retrieval_timing: RetrievalTimingOut
    generation_ms: float
    cost_usd: float | None


def _sources_out(sources: list[Source]) -> list[SourceOut]:
    return [
        SourceOut(
            marker=s.marker,
            document_id=s.document_id,
            filename=s.filename,
            page_from=s.page_from,
            page_to=s.page_to,
            section_path=s.section_path,
            heading=s.heading,
        )
        for s in sources
    ]


def _citations_out(citations: list[Citation]) -> list[CitationOut]:
    return [
        CitationOut(
            marker=c.marker,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            filename=c.filename,
            page_from=c.page_from,
            page_to=c.page_to,
            section_path=c.section_path,
            snippet=c.snippet,
        )
        for c in citations
    ]


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
        sources=_sources_out(result.sources),
        citations=_citations_out(result.citations),
        prompt_name=result.prompt_name,
        prompt_sha256=result.prompt_sha256,
        model=result.model,
        retrieval_timing=RetrievalTimingOut(**asdict(result.retrieval_timing)),
        generation_ms=result.generation_ms,
        cost_usd=result.cost_usd,
    )


_HEARTBEAT = StreamEvent("heartbeat", {})


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _event_payload(event: StreamEvent) -> dict[str, Any]:
    """Turn one `StreamEvent`'s data into JSON-ready values -- "sources"
    and "done" are the only two carrying domain objects (`Source` /
    `Citation`) rather than plain values already."""
    if event.event == "sources":
        return {"sources": [s.model_dump(mode="json") for s in _sources_out(event.data["sources"])]}
    if event.event == "done":
        payload = dict(event.data)
        payload["citations"] = [
            c.model_dump(mode="json") for c in _citations_out(event.data["citations"])
        ]
        return payload
    return event.data


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    session: DbSession,
    embedder: EmbedderDep,
    reranker: RerankerDep,
    llm: LlmClientDep,
) -> StreamingResponse:
    async def sse() -> AsyncIterator[str]:
        events = with_heartbeat(
            generate_answer_stream(
                session,
                embedder,
                reranker,
                llm,
                tenant_id=payload.tenant_id,
                question=payload.question,
                strategy=payload.strategy,
            ),
            interval=15.0,
            heartbeat=_HEARTBEAT,
        )
        try:
            async for event in events:
                if await request.is_disconnected():
                    _log.info("client_disconnected", question=payload.question)
                    break
                if event.event == "heartbeat":
                    yield ": heartbeat\n\n"
                    continue
                yield _sse(event.event, _event_payload(event))
        finally:
            # Cascades through with_heartbeat into generate_answer_stream
            # into the LlmClient's own stream -- one real network read
            # actually gets torn down, not just this generator.
            await events.aclose()

    return StreamingResponse(
        sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )
