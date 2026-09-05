"""``generate_answer`` (non-streaming) and ``generate_answer_stream``
(Giorno 14, SSE) -- both walk the same path::

    retrieve_context  ->  [pre-generation abstention gate, Giorno 13]
      ->  build_context  ->  detect version conflicts (Giorno 13)
      ->  render the versioned prompt  ->  LlmClient.complete / .stream
      ->  extract_and_validate (Giorno 12)  ->  answer + valid
      citations + the marker -> source map

``_prepare`` holds everything the two share, up to but not including the
LLM call itself, so the pre-generation gate, the conflict check and their
log events exist in exactly one place.

If the top reranked chunk's score is below `settings.
min_rerank_score_for_answer` (or nothing was retrieved at all), the LLM
is never called: the answer is NICHT_GEFUNDEN and the event is logged as
`abstained_pre_generation` -- the cost-saving path Giorno 13 asks for.
Past that gate, citation validation (`domain.citations.
extract_and_validate`) drops any marker the model invented and turns an
answer with claims but zero valid citations into an abstention -- never
a failed request. **For the streaming path this runs only after the full
answer has been assembled**, so a marker already sent to the client as a
`token` event cannot be un-sent if it later turns out to be invented;
the `done` event's `citations` is still the honest, validated list even
when it disagrees with what was streamed (see docs/FAILURE-MODES.md). A
retrieval whose sources disagree by version on the same requirement code
(`domain.conflicts.find_version_conflicts`) adds an explicit instruction
to the system message asking the model to flag the discrepancy and cite
both versions. Every answer logs which prompt produced it -- name and
content hash -- so it traces back to its exact instructions.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embedding.base import EmbeddingAdapter
from app.adapters.llm.base import LlmClient
from app.adapters.reranker.base import Reranker
from app.core.config import settings
from app.core.metrics import hallucinated_citation_total
from app.domain.citations import ABSTENTION_TEXT, Citation, extract_and_validate
from app.domain.conflicts import find_version_conflicts
from app.domain.context import Source, build_context
from app.domain.models import PipelineTiming, RetrievalResult
from app.services.prompts import Prompt, load_prompt
from app.services.retrieval import retrieve_context

# A short, fixed steer. The behavioural contract (German, cite only
# [S...], NICHT_GEFUNDEN, no outside knowledge) lives in the versioned
# prompt file, not here.
_SYSTEM = (
    "Du bist ein praeziser Assistent fuer technische Normen. "
    "Halte dich strikt an die folgenden Anweisungen."
)

# Appended to the system message only when find_version_conflicts finds
# more than one version_label among the retrieved sources for the same
# document (Giorno 13).
_CONFLICT_INSTRUCTION = (
    "Achtung: Einige Quellen stammen aus unterschiedlichen Versionen "
    "desselben Dokuments. Weise in der Antwort ausdruecklich auf die "
    "Abweichung hin und nenne beide Versionen mit ihren Markierungen."
)

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    sources: list[Source]
    citations: list[Citation]
    prompt_name: str
    prompt_sha256: str
    model: str
    retrieval_timing: PipelineTiming
    generation_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One SSE event (plan, Giorno 14): `event: {event}\\ndata:
    {json.dumps(data)}\\n\\n`. `data` may hold plain JSON-ready values, or
    (for "sources"/"done") the domain `Source`/`Citation` objects
    themselves -- the API layer, not this pure service, turns those into
    their wire shape (`api.v1.chat.SourceOut`/`CitationOut`), same split
    Day 11/12 already draw for the non-streaming response.
    """

    event: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Prepared:
    prompt: Prompt
    retrieval: RetrievalResult
    sources: dict[str, Source]
    system: str
    user_message: str
    abstain_pre_generation: bool


async def _prepare(
    session: AsyncSession,
    embedder: EmbeddingAdapter,
    reranker: Reranker,
    *,
    tenant_id: UUID,
    question: str,
    strategy: str | None,
    prompt_name: str | None,
) -> _Prepared:
    prompt = load_prompt(prompt_name or settings.answer_prompt_name)

    retrieval = await retrieve_context(
        session,
        embedder,
        reranker,
        tenant_id=tenant_id,
        question=question,
        strategy=strategy,
    )

    top_score = retrieval.reranked[0].score if retrieval.reranked else None
    if top_score is None or top_score < settings.min_rerank_score_for_answer:
        _log.info(
            "abstained_pre_generation",
            top_score=top_score,
            threshold=settings.min_rerank_score_for_answer,
            n_candidates=len(retrieval.reranked),
            retrieval_ms=retrieval.timing.total_ms,
        )
        return _Prepared(
            prompt=prompt,
            retrieval=retrieval,
            sources={},
            system=_SYSTEM,
            user_message="",
            abstain_pre_generation=True,
        )

    context_block, sources = build_context(retrieval.context)
    user_message = prompt.render(context=context_block, question=question)

    system = _SYSTEM
    conflicts = find_version_conflicts(sources)
    if conflicts:
        system = f"{system} {_CONFLICT_INSTRUCTION}"
        _log.info("version_conflict_detected", codes=list(conflicts))

    return _Prepared(
        prompt=prompt,
        retrieval=retrieval,
        sources=sources,
        system=system,
        user_message=user_message,
        abstain_pre_generation=False,
    )


async def generate_answer(
    session: AsyncSession,
    embedder: EmbeddingAdapter,
    reranker: Reranker,
    llm: LlmClient,
    *,
    tenant_id: UUID,
    question: str,
    strategy: str | None = None,
    prompt_name: str | None = None,
) -> AnswerResult:
    prep = await _prepare(
        session,
        embedder,
        reranker,
        tenant_id=tenant_id,
        question=question,
        strategy=strategy,
        prompt_name=prompt_name,
    )

    if prep.abstain_pre_generation:
        return AnswerResult(
            answer=ABSTENTION_TEXT,
            sources=[],
            citations=[],
            prompt_name=prep.prompt.name,
            prompt_sha256=prep.prompt.sha256,
            model=llm.name,
            retrieval_timing=prep.retrieval.timing,
            generation_ms=0.0,
            prompt_tokens=None,
            completion_tokens=None,
        )

    started = time.perf_counter()
    completion = await llm.complete(
        system=prep.system,
        user=prep.user_message,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    generation_ms = round((time.perf_counter() - started) * 1000, 1)

    answer, citations, invented_markers = extract_and_validate(completion.text, prep.sources)
    if invented_markers:
        hallucinated_citation_total.inc(len(invented_markers))
        _log.warning(
            "citation.invented_markers_dropped",
            markers=invented_markers,
            prompt_name=prep.prompt.name,
            model=completion.model,
        )

    _log.info(
        "answer.generated",
        prompt_name=prep.prompt.name,
        prompt_sha256=prep.prompt.sha256,
        model=completion.model,
        n_sources=len(prep.sources),
        n_citations=len(citations),
        retrieval_ms=prep.retrieval.timing.total_ms,
        generation_ms=generation_ms,
    )

    return AnswerResult(
        answer=answer,
        sources=list(prep.sources.values()),
        citations=citations,
        prompt_name=prep.prompt.name,
        prompt_sha256=prep.prompt.sha256,
        model=completion.model,
        retrieval_timing=prep.retrieval.timing,
        generation_ms=generation_ms,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
    )


async def generate_answer_stream(
    session: AsyncSession,
    embedder: EmbeddingAdapter,
    reranker: Reranker,
    llm: LlmClient,
    *,
    tenant_id: UUID,
    question: str,
    strategy: str | None = None,
    prompt_name: str | None = None,
) -> AsyncGenerator[StreamEvent]:
    """`stage(retrieving)` -> `stage(reranking)` -> `sources` (BEFORE any
    token -- the plan is explicit this is what makes the UI feel fast) ->
    `stage(generating)` -> `token` (repeated) -> `done`, or `error` in
    place of whatever would have come next. Any exception -- a typed
    `NormeonError` from the LLM adapter or anything else -- becomes one
    `error` event; by the time this is streaming, the HTTP response has
    already started, so main.py's generic handler can no longer turn it
    into a JSON body the way it does for the non-streaming endpoint.

    `cost_usd` is always `null` here -- cost tracking is Day 15; the key
    is already present so the wire shape does not change once it lands.
    """
    try:
        yield StreamEvent("stage", {"stage": "retrieving"})
        yield StreamEvent("stage", {"stage": "reranking"})

        prep = await _prepare(
            session,
            embedder,
            reranker,
            tenant_id=tenant_id,
            question=question,
            strategy=strategy,
            prompt_name=prompt_name,
        )

        if prep.abstain_pre_generation:
            yield StreamEvent("sources", {"sources": []})
            yield StreamEvent("token", {"text": ABSTENTION_TEXT})
            yield StreamEvent(
                "done",
                {
                    "latency_ms": {
                        "retrieval_ms": prep.retrieval.timing.total_ms,
                        "generation_ms": 0.0,
                        "total_ms": prep.retrieval.timing.total_ms,
                    },
                    "cost_usd": None,
                    "citations": [],
                },
            )
            return

        yield StreamEvent("sources", {"sources": list(prep.sources.values())})
        yield StreamEvent("stage", {"stage": "generating"})

        started = time.perf_counter()
        parts: list[str] = []
        token_stream = llm.stream(
            system=prep.system,
            user=prep.user_message,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        try:
            async for chunk in token_stream:
                parts.append(chunk)
                yield StreamEvent("token", {"text": chunk})
        finally:
            # Cascades to a caller closing *this* generator early (a
            # client disconnect, api.v1.chat) down to the adapter's own
            # network read -- "annulla il task LLM" from the plan.
            await token_stream.aclose()
        generation_ms = round((time.perf_counter() - started) * 1000, 1)

        answer, citations, invented_markers = extract_and_validate(
            "".join(parts), prep.sources
        )
        if invented_markers:
            hallucinated_citation_total.inc(len(invented_markers))
            _log.warning(
                "citation.invented_markers_dropped",
                markers=invented_markers,
                prompt_name=prep.prompt.name,
                model=llm.name,
            )

        _log.info(
            "answer.generated",
            prompt_name=prep.prompt.name,
            prompt_sha256=prep.prompt.sha256,
            model=llm.name,
            n_sources=len(prep.sources),
            n_citations=len(citations),
            retrieval_ms=prep.retrieval.timing.total_ms,
            generation_ms=generation_ms,
        )

        yield StreamEvent(
            "done",
            {
                "latency_ms": {
                    "retrieval_ms": prep.retrieval.timing.total_ms,
                    "generation_ms": generation_ms,
                    "total_ms": prep.retrieval.timing.total_ms + generation_ms,
                },
                "cost_usd": None,
                "citations": citations,
            },
        )
    except Exception as exc:  # deliberately broad: an SSE client must
        # never see a silently dead connection -- see the docstring.
        code = getattr(exc, "code", "internal_error")
        _log.warning("chat_stream.failed", code=code, error=str(exc))
        yield StreamEvent("error", {"code": code, "message": str(exc)})
