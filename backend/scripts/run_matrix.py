"""Run the Giorno 20 experiment matrix -- {fixed_500, structural} x
{vector, hybrid}, reranker on, top_k=5 -- and write a comparison report.

    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/run_matrix.py

For each of the 4 configurations: run_evaluation (real answer-generation
calls) -> write_report -> score_report (real judge calls) ->
write_scored_report -> compute_metrics. Writes each configuration's
usual `<hash>.json`/`.scored.json`/`.metrics.json` next to each other in
`eval/reports/`, then `eval/reports/matrix.md` and `matrix.html` with
all 4 side by side.

**Cost-bearing**: 4 configs x 50 questions = 200 real answer-generation
calls, then 200 more real judge calls -- needs a real (non-"fake")
LLM_PROVIDER/LLM_API_BASE_URL/LLM_MODEL configured, and is not free on a
rate-limited provider. Lower --concurrency if a run comes back with 429
Too Many Requests.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.adapters.reranker.cross_encoder import CrossEncoderReranker
from app.api.deps import get_embedder, get_llm_client
from app.core.config import settings
from app.db.seed import DEMO_TENANT_ID
from app.db.session import async_session_factory
from app.eval.dataset import DATASET_DIR, load_questions
from app.eval.matrix import MatrixRow, write_matrix_report
from app.eval.metrics import compute_metrics
from app.eval.models import EvalConfig, EvalQuestion
from app.eval.runner import run_evaluation, write_report
from app.eval.scoring import score_report, write_scored_report

_MATRIX = [
    ("fixed_500 / vector", "fixed_500", "vector"),
    ("fixed_500 / hybrid", "fixed_500", "hybrid"),
    ("structural / vector", "structural", "vector"),
    ("structural / hybrid", "structural", "hybrid"),
]


async def _run_one(
    label: str,
    chunking_strategy: str,
    retrieval_mode: str,
    *,
    questions: list[EvalQuestion],
    top_k: int,
    concurrency: int,
) -> MatrixRow:
    reranker = CrossEncoderReranker(settings.reranker_model)
    config = EvalConfig(
        chunking_strategy=chunking_strategy,  # type: ignore[arg-type]
        retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
        reranker="cross_encoder",
        top_k=top_k,
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
    )

    print(f"\n=== {label} ===", flush=True)
    report = await run_evaluation(
        questions,
        config,
        session_factory=async_session_factory,
        embedder=get_embedder(),
        reranker=reranker,
        llm=get_llm_client(),
        tenant_id=DEMO_TENANT_ID,
        concurrency=concurrency,
    )
    report_path = write_report(report)
    print(f"report -> {report_path}", flush=True)

    scored = await score_report(report, questions, get_llm_client())
    scored_path = write_scored_report(scored, reports_dir=report_path.parent)
    print(f"scored -> {scored_path}", flush=True)

    metrics = compute_metrics(report, scored, questions)
    print(metrics.model_dump_json(indent=2), flush=True)

    return MatrixRow(label=label, config=config, metrics=metrics)


async def _run(args: argparse.Namespace) -> None:
    questions = load_questions(Path(args.questions))
    rows = []
    for label, chunking_strategy, retrieval_mode in _MATRIX:
        row = await _run_one(
            label,
            chunking_strategy,
            retrieval_mode,
            questions=questions,
            top_k=args.top_k,
            concurrency=args.concurrency,
        )
        rows.append(row)

    md_path, html_path = write_matrix_report(rows)
    print(f"\nmatrix report -> {md_path}")
    print(f"matrix report -> {html_path}")


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions", default=str(DATASET_DIR / "questions.yaml")
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help=(
            "questions in flight at once per configuration (default: 2, "
            "lower than run_eval.py's default -- this script issues 4x "
            "the LLM traffic in one run). Lower further if a run comes "
            "back with 429 Too Many Requests."
        ),
    )
    asyncio.run(_run(parser.parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
