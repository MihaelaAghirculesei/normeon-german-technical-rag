"""Run the Giorno 28 ablation matrix and write a comparison report.

One baseline -- structural chunking, hybrid retrieval, cross-encoder
reranker, top_k=5, the configured model -- and one row per experiment that
changes exactly one of those knobs (`EXPERIMENTS`), so every difference
against the baseline has one cause. The judge is the configured model in
every row, including the one that changes the answering model.

E7 embeds with Gemini (`gemini-embedding-2` at 1024 dimensions) in place of
local e5; its vectors sit next to e5's in the same table, keyed by model, so
the corpus must be embedded with it first (`scripts/embed_missing.py`).
`deployed` is the configuration the demo runs (ADR 0005): Gemini
embeddings and no reranker -- two knobs, so it is a reference row, not an
ablation. Both need `EMBEDDING_API_KEY`.

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

**Cost-bearing**: up to 11 x 50 answer-generation calls (fewer: the
confidence gate abstains without one) plus 11 x 50 judge calls against the
configured (non-"fake") LLM. Use `--questions` with the
smoke set and `--reports-dir` pointing at a scratch directory for a
dry run first.
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.adapters.embedding.base import EmbeddingAdapter
from app.adapters.embedding.gemini import GeminiEmbedder
from app.adapters.llm.base import LlmClient
from app.adapters.reranker.base import Reranker
from app.adapters.reranker.caching import CachingReranker
from app.adapters.reranker.cross_encoder import CrossEncoderReranker
from app.adapters.reranker.noop import NoopReranker
from app.api.deps import build_llm_client, get_embedder
from app.core.config import settings
from app.db.seed import DEMO_TENANT_ID
from app.db.session import async_session_factory
from app.domain.models import FtsMatch, RetrievalMode
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


@dataclass(frozen=True)
class Experiment:
    """One row: the baseline with at most one knob changed. `model=None`
    is the configured model; `"B"` is the one given by `--model-b`."""

    label: str
    chunking: Literal["fixed_500", "structural"] = "structural"
    mode: RetrievalMode = "hybrid"
    rerank: bool = True
    top_k: int | None = None  # None: --top-k
    model: Literal["B"] | None = None
    fts_match: FtsMatch = "all"
    embedding: Literal["gemini"] | None = None  # None: the configured embedder


GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
# The first row is the baseline; every other row is compared against it
# question by question.
BASELINE = "baseline"
EXPERIMENTS = [
    Experiment(BASELINE),
    Experiment("E1 fixed_500 chunks", chunking="fixed_500"),
    Experiment("E2 vector only", mode="vector"),
    Experiment("E3 full-text only", mode="fts"),
    Experiment("E4 no reranker", rerank=False),
    Experiment("E5 top-k 3", top_k=3),
    Experiment("E6 top-k 10", top_k=10),
    Experiment("E7 Gemini embeddings", embedding="gemini"),
    Experiment("E8 model B", model="B"),
    Experiment("E9 full-text on any term", fts_match="any"),
]
# Rows that change more than one knob: shown for reference, not as an
# ablation (a difference from the baseline has several causes).
REFERENCES = [
    Experiment("deployed: Gemini embeddings, no reranker", rerank=False, embedding="gemini"),
]


def _gemini_embedder() -> EmbeddingAdapter:
    if not settings.embedding_api_key:
        raise SystemExit("EMBEDDING_API_KEY must be set for the Gemini embedding rows")
    return GeminiEmbedder(
        api_key=settings.embedding_api_key,
        model=GEMINI_EMBEDDING_MODEL,
        dim=settings.embedding_dim,
    )


def _covers(report: EvalReport, questions: list[EvalQuestion]) -> bool:
    return sorted(r.question_id for r in report.results) == sorted(q.id for q in questions)


async def _report_for(
    config: EvalConfig,
    questions: list[EvalQuestion],
    reranker: Reranker,
    llm: LlmClient,
    embedder: EmbeddingAdapter,
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
        embedder=embedder,
        reranker=reranker,
        llm=llm,
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
    judge: LlmClient,
    *,
    reports_dir: Path,
    judge_interval: float,
) -> ScoredReport:
    path = reports_dir / f"{report.config_hash}.scored.json"
    if not path.exists():
        scored = await score_report(report, questions, judge, min_interval_s=judge_interval)
    else:
        existing = ScoredReport.model_validate_json(path.read_text(encoding="utf-8"))
        scored, n_rejudged = await rescore_stale(
            report, existing, questions, judge, min_interval_s=judge_interval
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
    experiment: Experiment,
    *,
    questions: list[EvalQuestion],
    reranker: CachingReranker,
    judge: LlmClient,
    args: argparse.Namespace,
) -> MatrixRow:
    model = args.model_b if experiment.model == "B" else settings.llm_model
    config = EvalConfig(
        chunking_strategy=experiment.chunking,
        retrieval_mode=experiment.mode,
        reranker="cross_encoder" if experiment.rerank else "noop",
        top_k=experiment.top_k or args.top_k,
        rerank_candidates=args.candidates,
        # the cross-encoder's input cap; meaningless without it
        reranker_max_length=args.max_length if experiment.rerank else None,
        prompt_name=settings.answer_prompt_name,
        embedding_provider=experiment.embedding or settings.embedding_provider,
        llm_provider=settings.llm_provider,
        llm_model=model,
        # only when changed, so the default's hash matches the other rows'
        fts_match=experiment.fts_match if experiment.fts_match != "all" else None,
    )
    reports_dir = Path(args.reports_dir)
    label = experiment.label

    print(f"\n=== {label} ({config_hash(config)[:12]}) ===", flush=True)
    started = time.perf_counter()
    report = await _report_for(
        config,
        questions,
        reranker if experiment.rerank else NoopReranker(),
        judge if model == settings.llm_model else build_llm_client(model),
        _gemini_embedder() if experiment.embedding == "gemini" else get_embedder(),
        reports_dir=reports_dir,
        concurrency=args.concurrency,
        question_interval=args.question_interval,
    )
    scored = await _scored_for(
        report, questions, judge, reports_dir=reports_dir, judge_interval=args.judge_interval
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
    judge = build_llm_client()
    try:
        rows = [
            await _run_one(e, questions=questions, reranker=reranker, judge=judge, args=args)
            for e in [*EXPERIMENTS, *REFERENCES]
        ]
    finally:
        reranker.close()

    notes = [
        f"Every row changes one knob of the baseline: structural chunking, hybrid "
        f"retrieval, cross-encoder over {args.candidates} candidates, top-k "
        f"{args.top_k}, answers from {settings.llm_model}, prompt "
        f"{settings.answer_prompt_name}. E8 answers with {args.model_b}; every row "
        f"is judged by {settings.llm_model}.",
        "E4 keeps the fused order and its scores, so the confidence gate -- set "
        "for cross-encoder logits -- does not fire there.",
        "E3 uses the production full-text query, which needs every term of the "
        "question; E9 keeps hybrid retrieval and lets full-text match any term.",
        f"E7 embeds with {GEMINI_EMBEDDING_MODEL} in place of "
        f"{settings.embedding_model}. `deployed` is the demo's configuration "
        "(Gemini embeddings, no reranker): two knobs changed, a reference row.",
    ]
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
        "--model-b",
        default="gemini-flash-latest",
        help="the answering model of E8 (default: gemini-flash-latest)",
    )
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
