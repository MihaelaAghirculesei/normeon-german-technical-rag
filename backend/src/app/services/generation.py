"""``generate_answer`` -- the non-streaming answer path (plan, Giorno 11)::

    retrieve_context  ->  build_context  ->  render the versioned prompt
      ->  LlmClient.complete  ->  answer + the marker -> source map

What Day 11 does *not* do: it does not validate the model's citation
markers (Day 12) and it does not apply a confidence threshold before
calling the model (Day 13). It does record which prompt produced the
answer -- name and content hash -- so any logged answer traces back to
its exact instructions.
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

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    sources: list[Source]
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
    context_block, sources = build_context(retrieval.context)
    user_message = prompt.render(context=context_block, question=question)

    started = time.perf_counter()
    completion = await llm.complete(
        system=_SYSTEM,
        user=user_message,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    generation_ms = round((time.perf_counter() - started) * 1000, 1)

    _log.info(
        "answer.generated",
        prompt_name=prompt.name,
        prompt_sha256=prompt.sha256,
        model=completion.model,
        n_sources=len(sources),
        retrieval_ms=retrieval.timing.total_ms,
        generation_ms=generation_ms,
    )

    return AnswerResult(
        answer=completion.text,
        sources=list(sources.values()),
        prompt_name=prompt.name,
        prompt_sha256=prompt.sha256,
        model=completion.model,
        retrieval_timing=retrieval.timing,
        generation_ms=generation_ms,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
    )
