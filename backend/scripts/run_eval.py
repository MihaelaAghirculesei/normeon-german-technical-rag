"""Run an evaluation question set against the dev corpus and write
`backend/eval/reports/<config_hash>.json` (plan, Giorno 16).

    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/run_eval.py

Defaults to the throwaway smoke set (`eval/dataset/questions.smoke.yaml`)
and the offline `fake` LLM. The real 50-question set (Giorni 17-18) is
passed with `--questions eval/dataset/questions.yaml`; point at a real
model with the usual `LLM_PROVIDER`/`LLM_API_BASE_URL`/`LLM_MODEL` env
vars.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.adapters.reranker.cross_encoder import CrossEncoderReranker
from app.adapters.reranker.noop import NoopReranker
from app.api.deps import get_embedder, get_llm_client
from app.core.config import settings
from app.db.seed import DEMO_TENANT_ID
from app.db.session import async_session_factory
from app.eval.dataset import SMOKE_QUESTIONS, load_questions
from app.eval.models import EvalConfig
from app.eval.runner import run_evaluation, write_report


async def _run(args: argparse.Namespace) -> None:
    questions = load_questions(Path(args.questions))
    reranker = (
        CrossEncoderReranker(settings.reranker_model)
        if args.reranker == "cross_encoder"
        else NoopReranker()
    )
    config = EvalConfig(
        chunking_strategy=args.strategy,
        reranker=args.reranker,
        top_k=args.top_k,
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
    )

    report = await run_evaluation(
        questions,
        config,
        session_factory=async_session_factory,
        embedder=get_embedder(),
        reranker=reranker,
        llm=get_llm_client(),
        tenant_id=DEMO_TENANT_ID,
    )
    path = write_report(report)

    abstained = sum(1 for r in report.results if r.abstained)
    errored = sum(1 for r in report.results if r.error)
    print(f"\n{report.n_questions} questions | {abstained} abstained | {errored} errored")
    print(f"config_hash {report.config_hash}")
    print(f"report      {path}")


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(SMOKE_QUESTIONS))
    parser.add_argument("--strategy", choices=["fixed_500", "structural"], default="structural")
    parser.add_argument("--reranker", choices=["noop", "cross_encoder"], default="noop")
    parser.add_argument("--top-k", type=int, default=5)
    asyncio.run(_run(parser.parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
