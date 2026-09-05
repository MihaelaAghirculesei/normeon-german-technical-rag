"""``generate_answer`` -- the non-streaming answer path::

    retrieve_context  ->  [pre-generation abstention gate, Giorno 13]
      ->  build_context  ->  detect version conflicts (Giorno 13)
      ->  render the versioned prompt  ->  LlmClient.complete
      ->  extract_and_validate (Giorno 12)  ->  answer + valid
      citations + the marker -> source map

If the top reranked chunk's score is below `settings.
min_rerank_score_for_answer` (or nothing was retrieved at all), the LLM
is never called: the answer is NICHT_GEFUNDEN and the event is logged as
`abstained_pre_generation` -- the cost-saving path Giorno 13 asks for.
Past that gate, citation validation (`domain.citations.
extract_and_validate`) drops any marker the model invented and turns an
answer with claims but zero valid citations into an abstention -- never
a failed request. A retrieval whose sources disagree by version on the
same requirement code (`domain.conflicts.find_version_conflicts`) adds
an explicit instruction to the system message asking the model to flag
the discrepancy and cite both versions. It records which prompt produced
the answer -- name and content hash -- so any logged answer traces back
to its exact instructions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
from app.domain.models import PipelineTiming
from app.services.prompts import load_prompt
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
        return AnswerResult(
            answer=ABSTENTION_TEXT,
            sources=[],
            citations=[],
            prompt_name=prompt.name,
            prompt_sha256=prompt.sha256,
            model=llm.name,
            retrieval_timing=retrieval.timing,
            generation_ms=0.0,
            prompt_tokens=None,
            completion_tokens=None,
        )

    context_block, sources = build_context(retrieval.context)
    user_message = prompt.render(context=context_block, question=question)

    system = _SYSTEM
    conflicts = find_version_conflicts(sources)
    if conflicts:
        system = f"{system} {_CONFLICT_INSTRUCTION}"
        _log.info("version_conflict_detected", codes=list(conflicts))

    started = time.perf_counter()
    completion = await llm.complete(
        system=system,
        user=user_message,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    generation_ms = round((time.perf_counter() - started) * 1000, 1)

    answer, citations, invented_markers = extract_and_validate(completion.text, sources)
    if invented_markers:
        hallucinated_citation_total.inc(len(invented_markers))
        _log.warning(
            "citation.invented_markers_dropped",
            markers=invented_markers,
            prompt_name=prompt.name,
            model=completion.model,
        )

    _log.info(
        "answer.generated",
        prompt_name=prompt.name,
        prompt_sha256=prompt.sha256,
        model=completion.model,
        n_sources=len(sources),
        n_citations=len(citations),
        retrieval_ms=retrieval.timing.total_ms,
        generation_ms=generation_ms,
    )

    return AnswerResult(
        answer=answer,
        sources=list(sources.values()),
        citations=citations,
        prompt_name=prompt.name,
        prompt_sha256=prompt.sha256,
        model=completion.model,
        retrieval_timing=retrieval.timing,
        generation_ms=generation_ms,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
    )
