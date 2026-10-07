"""Run a question set through the real answer path under one
configuration and write `backend/eval/reports/<config_hash>.json`
(plan, Giorno 16).

Scoring against `gold_sources` / `expected_answer_points` is NOT here --
this only captures raw run output (answer, citations, retrieved chunks,
latency, cost). The metrics that read this report are Giorno 19.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
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
    # `exclude_none`: a knob added to EvalConfig later (default None)
    # leaves the hash of every configuration that doesn't set it -- and
    # so every report already on disk -- unchanged.
    return compute_config_hash(**config.model_dump(exclude_none=True))


def question_set_fingerprint(questions: list[EvalQuestion]) -> str:
    """Short hash of the questions' ids *and* text: two sets that reuse
    the same ids (the smoke set and the real one both start at Q001) must
    never share a checkpoint."""
    payload = json.dumps([[q.id, q.question] for q in questions], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def checkpoint_path(
    config: EvalConfig, questions: list[EvalQuestion], reports_dir: Path = REPORTS_DIR
) -> Path:
    """Where `run_evaluation` appends each finished question while a run
    is in progress, so a run that dies can resume instead of starting
    over from question 1. Keyed by configuration and question set."""
    name = f"{config_hash(config)}.{question_set_fingerprint(questions)}.partial.jsonl"
    return reports_dir / name


def _load_checkpoint(path: Path, wanted: set[str]) -> dict[str, QuestionRun]:
    """Completed runs from a previous attempt, keyed by question id. An
    errored run is not kept -- resuming retries it -- and an unparsable
    line (the process died mid-write) is skipped. A later line for the
    same question wins."""
    if not path.exists():
        return {}
    done: dict[str, QuestionRun] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            run = QuestionRun.model_validate_json(line)
        except ValueError:
            continue
        if run.question_id not in wanted:
            continue
        if run.error is None:
            done[run.question_id] = run
        else:
            done.pop(run.question_id, None)
    return done


def _append_checkpoint(path: Path, run: QuestionRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(run.model_dump_json() + "\n")


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
                    retrieval_mode=config.retrieval_mode,
                    rerank_top_k=config.top_k,
                    candidate_k=config.rerank_candidates,
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
        invented_citations=result.invented_citations,
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
    checkpoint: Path | None = None,
    min_interval_s: float = 0.0,
) -> EvalReport:
    """With `checkpoint`, every finished question is appended to that
    file as it completes, and questions a previous attempt already
    completed (without error) are taken from it instead of being run
    again. The caller removes the file once the final report is
    written (see `checkpoint_path`).

    `min_interval_s` spaces question starts at least that far apart. When
    retrieval is fast (cached rerank scores) the LLM calls would otherwise
    go out back to back and trip a requests-per-minute quota."""
    run_hash = config_hash(config)
    sem = asyncio.Semaphore(concurrency)
    done = _load_checkpoint(checkpoint, {q.id for q in questions}) if checkpoint else {}
    if done:
        _log.info("eval.resumed", config_hash=run_hash, already_done=len(done))
    finished = len(done)
    pace = asyncio.Lock()
    last_start: float | None = None

    async def wait_for_slot() -> None:
        nonlocal last_start
        async with pace:
            if last_start is not None:
                await asyncio.sleep(max(0.0, last_start + min_interval_s - time.monotonic()))
            last_start = time.monotonic()

    async def run_and_record(question: EvalQuestion) -> QuestionRun:
        nonlocal finished
        if question.id in done:
            return done[question.id]
        if min_interval_s > 0:
            await wait_for_slot()
        run = await _run_one(
            question,
            config,
            run_hash,
            session_factory=session_factory,
            embedder=embedder,
            reranker=reranker,
            llm=llm,
            tenant_id=tenant_id,
            sem=sem,
        )
        # Appends happen on the event loop thread between awaits, so two
        # questions finishing together can't interleave their lines.
        if checkpoint is not None:
            _append_checkpoint(checkpoint, run)
        finished += 1
        _log.info(
            "eval.question_done",
            question_id=question.id,
            progress=f"{finished}/{len(questions)}",
            total_ms=run.total_ms,
            error=run.error,
        )
        return run

    results = await asyncio.gather(*(run_and_record(q) for q in questions))
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
