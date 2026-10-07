"""Retrieval-only evaluation: recall@k, MRR and precision@k for the
Giorno 20 matrix configurations, without a single LLM call.

    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/run_retrieval_eval.py

Retrieval metrics only depend on which chunks reach the context, so they
can be measured in minutes, for free, and independently of any provider
quota -- useful to check a retrieval change before paying for a full
answer-and-judge run. Uses the same pipeline, candidate budget and
reranker (behind the same persistent score cache) as run_matrix.py, and
the same gold-matching rules as the full metrics.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.adapters.reranker.caching import CachingReranker
from app.adapters.reranker.cross_encoder import CrossEncoderReranker
from app.api.deps import get_embedder
from app.core.config import settings
from app.db.seed import DEMO_TENANT_ID
from app.db.session import async_session_factory
from app.eval.dataset import DATASET_DIR, load_questions
from app.eval.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k
from app.eval.models import EvalQuestion, QuestionRun, RetrievedRecord
from app.eval.runner import REPORTS_DIR
from app.eval.stats import bootstrap_ci
from app.services.retrieval import retrieve_context

_MATRIX = [
    ("structural / hybrid", "structural", "hybrid"),
    ("structural / vector", "structural", "vector"),
    ("fixed_500 / hybrid", "fixed_500", "hybrid"),
    ("fixed_500 / vector", "fixed_500", "vector"),
]


async def _retrieved(
    question: EvalQuestion,
    strategy: str,
    mode: str,
    reranker: CachingReranker,
    args: argparse.Namespace,
) -> QuestionRun:
    async with async_session_factory() as session:
        result = await retrieve_context(
            session,
            get_embedder(),
            reranker,
            tenant_id=DEMO_TENANT_ID,
            question=question.question,
            strategy=strategy,
            retrieval_mode=mode,  # type: ignore[arg-type]
            candidate_k=args.candidates,
            rerank_top_k=args.top_k,
        )
    # Only the fields the retrieval metrics read are meaningful here.
    return QuestionRun(
        question_id=question.id,
        category=question.category,
        answer="",
        abstained=False,
        citations=[],
        retrieved=[
            RetrievedRecord(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                filename=c.filename,
                page_from=c.page_from,
                page_to=c.page_to,
                section_path=c.section_path,
            )
            for c in result.context
        ],
        retrieval_ms=result.timing.total_ms,
        generation_ms=0.0,
        total_ms=result.timing.total_ms,
        cost_usd=None,
        prompt_name="",
        prompt_sha256="",
        model="",
    )


def _format_ci(ci: tuple[float, float] | None) -> str:
    return "–" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


async def _run(args: argparse.Namespace) -> None:
    questions = load_questions(Path(args.questions))
    gold_bearing = [q for q in questions if q.gold_sources]
    reranker = CachingReranker(
        CrossEncoderReranker(settings.reranker_model, args.max_length),
        Path(args.reports_dir) / "rerank_cache.sqlite3",
        namespace=f"{settings.reranker_model}|max_length={args.max_length}",
    )
    print("| Config | recall_at_k | 95% CI | mrr | precision_at_k |")
    print("| --- | --- | --- | --- | --- |")
    try:
        for label, strategy, mode in _MATRIX:
            runs = [await _retrieved(q, strategy, mode, reranker, args) for q in gold_bearing]
            recall = recall_at_k(gold_bearing, runs)
            pairs = zip(gold_bearing, runs, strict=True)
            hits = [float(recall_at_k([q], [run]) == 1.0) for q, run in pairs]
            mrr = mean_reciprocal_rank(gold_bearing, runs)
            precision = precision_at_k(gold_bearing, runs)
            print(
                f"| {label} | {recall:.3f} | {_format_ci(bootstrap_ci(hits))} "
                f"| {mrr:.3f} | {precision:.3f} |",
                flush=True,
            )
    finally:
        reranker.close()


def main() -> int:
    sys.stdout.reconfigure(errors="replace", line_buffering=True)  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", default=str(DATASET_DIR / "questions.yaml"))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=settings.reranker_max_length)
    asyncio.run(_run(parser.parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
