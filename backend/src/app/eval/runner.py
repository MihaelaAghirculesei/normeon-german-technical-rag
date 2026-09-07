"""Run a question set through the real answer path under one
configuration and write `backend/eval/reports/<config_hash>.json`
(plan, Giorno 16).

Scoring against `gold_sources` / `expected_answer_points` is NOT here --
this only captures raw run output (answer, citations, retrieved chunks,
latency, cost). The metrics that read this report are Giorno 19.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.embedding.base import EmbeddingAdapter
from app.adapters.llm.base import LlmClient
from app.adapters.reranker.base import Reranker
from app.domain.citations import ABSTENTION_TEXT
from app.domain.config_hash import compute_config_hash
from app.eval.models import (
    CitationRecord,
    EvalConfig,
    EvalQuestion,
    EvalReport,
    QuestionRun,
    RetrievedRecord,
)
from app.services.generation import generate_answer

REPORTS_DIR = Path(__file__).parents[3] / "eval" / "reports"

_log = structlog.get_logger(__name__)


def config_hash(config: EvalConfig) -> str:
    return compute_config_hash(**config.model_dump())


async def _run_one(
    question: EvalQuestion,
    config: EvalConfig,
    run_hash: str,
    *,
    session_factory: async_sessionmaker[AsyncSession],
    embedder: EmbeddingAdapter,
    reranker: Reranker,
    llm: LlmClient,
    tenant_id: UUID,
    sem: asyncio.Semaphore,
) -> QuestionRun:
    async with sem:
        try:
            async with session_factory() as session:
                result = await generate_answer(
                    session,
                    embedder,
                    reranker,
                    llm,
                    tenant_id=tenant_id,
                    question=question.question,
                    strategy=config.chunking_strategy,
                    request_id=f"eval-{run_hash[:12]}-{question.id}",
                )
        except Exception as exc:  # one bad question must not sink the whole run
            _log.warning("eval.question_failed", question_id=question.id, error=str(exc))
            return QuestionRun(
                question_id=question.id,
                category=question.category,
                answer="",
                abstained=False,
                citations=[],
                retrieved=[],
                retrieval_ms=0.0,
                generation_ms=0.0,
                total_ms=0.0,
                cost_usd=None,
                prompt_name="",
                prompt_sha256="",
                model="",
                error=str(exc),
            )

    timing = result.retrieval_timing
    return QuestionRun(
        question_id=question.id,
        category=question.category,
        answer=result.answer,
        abstained=result.answer == ABSTENTION_TEXT,
        citations=[
            CitationRecord(
                marker=c.marker,
                document_id=c.document_id,
                filename=c.filename,
                page_from=c.page_from,
                page_to=c.page_to,
                section_path=c.section_path,
            )
            for c in result.citations
        ],
        retrieved=[
            RetrievedRecord(
                chunk_id=s.chunk_id,
                document_id=s.document_id,
                filename=s.filename,
                page_from=s.page_from,
                page_to=s.page_to,
                section_path=s.section_path,
            )
            for s in result.sources
        ],
        retrieval_ms=timing.total_ms,
        generation_ms=result.generation_ms,
        total_ms=round(timing.total_ms + result.generation_ms, 1),
        cost_usd=result.cost_usd,
        prompt_name=result.prompt_name,
        prompt_sha256=result.prompt_sha256,
        model=result.model,
    )


async def run_evaluation(
    questions: list[EvalQuestion],
    config: EvalConfig,
    *,
    session_factory: async_sessionmaker[AsyncSession],
    embedder: EmbeddingAdapter,
    reranker: Reranker,
    llm: LlmClient,
    tenant_id: UUID,
    concurrency: int = 4,
) -> EvalReport:
    run_hash = config_hash(config)
    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *(
            _run_one(
                q,
                config,
                run_hash,
                session_factory=session_factory,
                embedder=embedder,
                reranker=reranker,
                llm=llm,
                tenant_id=tenant_id,
                sem=sem,
            )
            for q in questions
        )
    )
    return EvalReport(
        config=config,
        config_hash=run_hash,
        created_at=datetime.now(UTC),
        n_questions=len(questions),
        results=list(results),
    )


def write_report(report: EvalReport, reports_dir: Path = REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{report.config_hash}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path
