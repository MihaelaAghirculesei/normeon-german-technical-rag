"""Run the Giorno 20 experiment matrix -- {fixed_500, structural} x
{vector, hybrid}, cross-encoder reranker, top_k=5 -- and write a
comparison report.

    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/run_matrix.py

For each configuration: run_evaluation (real answer-generation calls)
-> write_report -> score_report (real judge calls) -> write_scored_report
-> compute_metrics -> `<hash>.metrics.json`; then all of them into
`matrix.md` / `matrix.html`.

**Resumable at every level** -- on this CPU one configuration takes
hours, and a run that dies must not start over:

- every finished question is appended to a `.partial.jsonl` checkpoint,
  so a killed run picks up at the next unanswered question;
- a configuration whose report already exists is not re-run, and one
  already scored is not re-scored -- except its judge errors (a 429, a
  truncated response), which are retried;
- every cross-encoder score is cached in `rerank_cache.sqlite3`, so the
  vector and hybrid runs over one chunking strategy -- whose candidates
  overlap heavily -- only pay for the pairs they don't share.

Just run the same command again after a crash.

**Controlled comparison**: `--candidates` is how many chunks reach the
reranker in *both* retrieval modes (hybrid keeps that many after fusion),
so vector vs. hybrid differ only in which chunks fill the budget, not in
how many. It also bounds the rerank cost, linear in it on CPU.

**Cost-bearing**: 4 x 50 answer-generation calls plus 4 x 50 judge calls
against the configured (non-"fake") LLM. Use `--questions` with the
smoke set and `--reports-dir` pointing at a scratch directory for a
dry run first.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from app.adapters.reranker.caching import CachingReranker
from app.adapters.reranker.cross_encoder import CrossEncoderReranker
from app.api.deps import get_embedder, get_llm_client
from app.core.config import settings
from app.db.seed import DEMO_TENANT_ID
from app.db.session import async_session_factory
from app.eval.dataset import DATASET_DIR, load_questions
from app.eval.matrix import MatrixRow, write_matrix_report
from app.eval.metrics import EvalMetrics, compute_metrics
from app.eval.models import EvalConfig, EvalQuestion, EvalReport
from app.eval.runner import REPORTS_DIR, checkpoint_path, config_hash, run_evaluation, write_report
from app.eval.scoring import (
    JudgeError,
    ScoredReport,
    rescore_stale,
    score_report,
    write_scored_report,
)

# (label, chunking_strategy, retrieval_mode). The first row is the
# plan's ablation baseline (structural chunking + hybrid retrieval);
# every other row is compared against it question by question.
BASELINE = "structural / hybrid"
_MATRIX = [
    (BASELINE, "structural", "hybrid"),
    ("structural / vector", "structural", "vector"),
    ("fixed_500 / hybrid", "fixed_500", "hybrid"),
    ("fixed_500 / vector", "fixed_500", "vector"),
]


def _covers(report: EvalReport, questions: list[EvalQuestion]) -> bool:
    return sorted(r.question_id for r in report.results) == sorted(q.id for q in questions)


async def _report_for(
    config: EvalConfig,
    questions: list[EvalQuestion],
    reranker: CachingReranker,
    *,
    reports_dir: Path,
    concurrency: int,
    question_interval: float,
) -> EvalReport:
    path = reports_dir / f"{config_hash(config)}.json"
    if path.exists():
        report = EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
        if _covers(report, questions) and not any(r.error for r in report.results):
            print(f"report exists, reusing -> {path}", flush=True)
            return report
        print(f"report at {path} is incomplete or for another question set; re-running")

    checkpoint = checkpoint_path(config, questions, reports_dir)
    report = await run_evaluation(
        questions,
        config,
        session_factory=async_session_factory,
        embedder=get_embedder(),
        reranker=reranker,
        llm=get_llm_client(),
        tenant_id=DEMO_TENANT_ID,
        concurrency=concurrency,
        checkpoint=checkpoint,
        min_interval_s=question_interval,
    )
    written = write_report(report, reports_dir)
    n_errors = sum(1 for r in report.results if r.error)
    if n_errors:
        # keep the checkpoint: re-running retries exactly the failed ones
        print(f"WARNING: {n_errors} question(s) errored; re-run to retry them", flush=True)
    else:
        checkpoint.unlink(missing_ok=True)
    print(f"report -> {written}", flush=True)
    return report


async def _scored_for(
    report: EvalReport,
    questions: list[EvalQuestion],
    *,
    reports_dir: Path,
    judge_interval: float,
) -> ScoredReport:
    path = reports_dir / f"{report.config_hash}.scored.json"
    if not path.exists():
        scored = await score_report(
            report, questions, get_llm_client(), min_interval_s=judge_interval
        )
    else:
        existing = ScoredReport.model_validate_json(path.read_text(encoding="utf-8"))
        scored, n_rejudged = await rescore_stale(
            report, existing, questions, get_llm_client(), min_interval_s=judge_interval
        )
        if not n_rejudged:
            print(f"scored report exists, reusing -> {path}", flush=True)
            return scored
        print(f"re-judged {n_rejudged} judge error(s) or changed answer(s)", flush=True)
    write_scored_report(scored, reports_dir)
    n_judge_errors = sum(1 for r in scored.results if isinstance(r.layer2, JudgeError))
    if n_judge_errors:
        print(
            f"WARNING: {n_judge_errors} judge error(s) -- answer_accuracy falls back to "
            "Layer 1 for those; re-run to retry them",
            flush=True,
        )
    return scored


async def _run_one(
    label: str,
    chunking_strategy: str,
    retrieval_mode: str,
    *,
    questions: list[EvalQuestion],
    reranker: CachingReranker,
    args: argparse.Namespace,
) -> MatrixRow:
    config = EvalConfig(
        chunking_strategy=chunking_strategy,  # type: ignore[arg-type]
        retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
        reranker="cross_encoder",
        top_k=args.top_k,
        rerank_candidates=args.candidates,
        reranker_max_length=args.max_length,
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
    )
    reports_dir = Path(args.reports_dir)

    print(f"\n=== {label} ({config_hash(config)[:12]}) ===", flush=True)
    started = time.perf_counter()
    report = await _report_for(
        config,
        questions,
        reranker,
        reports_dir=reports_dir,
        concurrency=args.concurrency,
        question_interval=args.question_interval,
    )
    scored = await _scored_for(
        report, questions, reports_dir=reports_dir, judge_interval=args.judge_interval
    )
    metrics: EvalMetrics = compute_metrics(report, scored, questions)
    metrics_path = reports_dir / f"{report.config_hash}.metrics.json"
    metrics_path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"answer_accuracy={metrics.answer_accuracy:.3f} recall_at_k={metrics.recall_at_k} "
        f"({time.perf_counter() - started:.0f}s) -> {metrics_path}",
        flush=True,
    )
    return MatrixRow(label=label, config=config, metrics=metrics)


async def _run(args: argparse.Namespace) -> None:
    questions = load_questions(Path(args.questions))
    reports_dir = Path(args.reports_dir)
    # One model load for the whole matrix, behind a persistent score
    # cache. The namespace carries everything a score depends on besides
    # the (query, chunk) pair itself.
    # Without the shared cache, a fresh throwaway file: every pair is
    # scored, so latency is comparable across configurations.
    cache_file = reports_dir / (
        "rerank_cache.sqlite3" if args.rerank_cache else "rerank_cache.cold.sqlite3"
    )
    if not args.rerank_cache:
        cache_file.unlink(missing_ok=True)
    reranker = CachingReranker(
        CrossEncoderReranker(settings.reranker_model, args.max_length),
        cache_file,
        namespace=f"{settings.reranker_model}|max_length={args.max_length}",
    )
    try:
        rows = [
            await _run_one(
                label, chunking, mode, questions=questions, reranker=reranker, args=args
            )
            for label, chunking, mode in _MATRIX
        ]
    finally:
        reranker.close()

    notes = []
    if args.rerank_cache:
        notes.append(
            "Latency columns include rerank-cache hits: a configuration whose "
            "candidates an earlier one already scored (vector after hybrid over the "
            "same chunking) looks faster than it is. Quality metrics are unaffected "
            "-- a cached score is the same number. For latency, compare a run made "
            "with --no-rerank-cache."
        )
    md_path, html_path = write_matrix_report(
        rows, reports_dir, baseline_label=BASELINE, notes=notes
    )
    print(f"\nmatrix report -> {md_path}")
    print(f"matrix report -> {html_path}")


def main() -> int:
    sys.stdout.reconfigure(errors="replace", line_buffering=True)  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", default=str(DATASET_DIR / "questions.yaml"))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidates",
        type=int,
        default=10,
        help="chunks handed to the reranker in either retrieval mode (default: 10)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=settings.reranker_max_length,
        help="reranker token cap per (query, chunk) pair (default: from settings)",
    )
    parser.add_argument(
        "--question-interval",
        type=float,
        default=4.0,
        help=(
            "minimum seconds between question starts (default: 4, i.e. 15/min). "
            "Only binds when retrieval is fast, e.g. on rerank-cache hits"
        ),
    )
    parser.add_argument(
        "--judge-interval",
        type=float,
        default=5.0,
        help=(
            "minimum seconds between judge calls (default: 5, i.e. 12/min -- "
            "under Gemini's free-tier requests-per-minute quota)"
        ),
    )
    parser.add_argument(
        "--no-rerank-cache",
        dest="rerank_cache",
        action="store_false",
        help=(
            "score every (query, chunk) pair fresh, for latency numbers that are "
            "comparable across configurations (much slower)"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "questions in flight at once (default: 1). The cross-encoder is "
            "CPU-bound, so more only helps when the LLM, not the reranker, is "
            "the bottleneck -- and it multiplies memory pressure on a small box."
        ),
    )
    asyncio.run(_run(parser.parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
