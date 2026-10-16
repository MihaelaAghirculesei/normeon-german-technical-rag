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

from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Annotated, Literal
from uuid import UUID

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, RootModel

from app.api.deps import AuthDep, EmbedderDep, LlmClientDep, RerankerDep, TenantSession
from app.core.streaming import with_heartbeat
from app.domain.citations import Citation
from app.domain.context import Source
from app.services.generation import StreamEvent, generate_answer, generate_answer_stream

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
_log = structlog.get_logger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    strategy: Literal["fixed_500", "structural"] | None = None


class SourceOut(BaseModel):
    marker: str
    chunk_id: UUID
    document_id: UUID
    filename: str
    page_from: int
    page_to: int
    section_path: str | None
    heading: str | None
    # The text was recognised by OCR from a scanned page and may hold
    # recognition errors; the UI says so next to the source.
    ocr: bool


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
            chunk_id=s.chunk_id,
            document_id=s.document_id,
            filename=s.filename,
            page_from=s.page_from,
            page_to=s.page_to,
            section_path=s.section_path,
            heading=s.heading,
            ocr=s.ocr,
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
    auth: AuthDep,
    session: TenantSession,
    embedder: EmbedderDep,
    reranker: RerankerDep,
    llm: LlmClientDep,
) -> ChatResponse:
    result = await generate_answer(
        session,
        embedder,
        reranker,
        llm,
        tenant_id=auth.tenant_id,
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


class _EventStream(StreamingResponse):
    # Sets the media type FastAPI documents the 200 response under, so the
    # ChatStreamEvent schema lands on text/event-stream, not application/json.
    media_type = "text/event-stream"


class StageData(BaseModel):
    stage: Literal["retrieving", "reranking", "generating"]


class SourcesData(BaseModel):
    sources: list[SourceOut]


class TokenData(BaseModel):
    text: str


class StreamLatencyOut(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class DoneData(BaseModel):
    latency_ms: StreamLatencyOut
    cost_usd: float | None
    # The validated answer; it replaces the text streamed as `token`s.
    answer: str
    citations: list[CitationOut]


class ErrorData(BaseModel):
    code: str
    message: str


class StageEvent(BaseModel):
    event: Literal["stage"]
    data: StageData


class SourcesEvent(BaseModel):
    event: Literal["sources"]
    data: SourcesData


class TokenEvent(BaseModel):
    event: Literal["token"]
    data: TokenData


class DoneEvent(BaseModel):
    event: Literal["done"]
    data: DoneData


class ErrorEvent(BaseModel):
    event: Literal["error"]
    data: ErrorData


class ChatStreamEvent(
    RootModel[
        Annotated[
            StageEvent | SourcesEvent | TokenEvent | DoneEvent | ErrorEvent,
            Field(discriminator="event"),
        ]
    ]
):
    """One SSE frame of `/chat/stream`: `event:` is the discriminator,
    `data:` the JSON payload. Never sent as a JSON body -- it exists so the
    frame shapes are part of the OpenAPI contract the frontend is
    generated from."""


_HEARTBEAT = StreamEvent("heartbeat", {})


def _sse(event: str, data: BaseModel) -> str:
    return f"event: {event}\ndata: {data.model_dump_json()}\n\n"


def _event_payload(event: StreamEvent) -> BaseModel:
    """Turn one `StreamEvent` into its typed payload -- validated against
    the same models the OpenAPI document advertises."""
    if event.event == "stage":
        return StageData(**event.data)
    if event.event == "sources":
        return SourcesData(sources=_sources_out(event.data["sources"]))
    if event.event == "token":
        return TokenData(**event.data)
    if event.event == "done":
        return DoneData(
            latency_ms=StreamLatencyOut(**event.data["latency_ms"]),
            cost_usd=event.data["cost_usd"],
            answer=event.data["answer"],
            citations=_citations_out(event.data["citations"]),
        )
    return ErrorData(**event.data)


@router.post(
    "/stream",
    response_class=_EventStream,
    responses={
        200: {
            "model": ChatStreamEvent,
            "description": "A `text/event-stream` of `ChatStreamEvent` frames.",
            "content": {"text/event-stream": {}},
        }
    },
)
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    auth: AuthDep,
    session: TenantSession,
    embedder: EmbedderDep,
    reranker: RerankerDep,
    llm: LlmClientDep,
) -> _EventStream:
    async def sse() -> AsyncIterator[str]:
        events = with_heartbeat(
            generate_answer_stream(
                session,
                embedder,
                reranker,
                llm,
                tenant_id=auth.tenant_id,
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

    return _EventStream(sse(), headers={"Cache-Control": "no-cache"})
