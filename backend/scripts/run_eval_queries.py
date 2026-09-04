"""Run every question in eval/retrieval_queries.txt through the full
retrieval pipeline and print a table. Ancestor of the Week 4 evaluation
suite -- no scoring yet, just "what does the pipeline return, and how
long did each phase take".

Uses the noop reranker by default (fast, deterministic); pass
`--rerank cross_encoder` for the real cross-encoder, which is ~10 s per
candidate pair on CPU (see scripts/try_rerank.py) so expect minutes.

Run with:
    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/run_eval_queries.py
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.adapters.reranker.base import Reranker
from app.adapters.reranker.cross_encoder import CrossEncoderReranker
from app.adapters.reranker.noop import NoopReranker
from app.api.deps import get_embedder
from app.core.config import settings
from app.db.seed import DEMO_TENANT_ID
from app.db.session import async_session_factory
from app.services.retrieval import retrieve_context

QUERIES_FILE = Path(__file__).parents[1] / "eval" / "retrieval_queries.txt"


def _load_queries() -> list[str]:
    lines = QUERIES_FILE.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _make_reranker(kind: str) -> Reranker:
    if kind == "cross_encoder":
        return CrossEncoderReranker(settings.reranker_model)
    return NoopReranker()


async def _run(strategy: str | None, rerank_kind: str) -> None:
    questions = _load_queries()
    embedder = get_embedder()
    reranker = _make_reranker(rerank_kind)
    await asyncio.to_thread(embedder.embed_query, "warmup")

    header = (
        f"{'#':>2}  {'question':<52}  {'ctx':>3}  {'top hit (file / section)':<34}  "
        f"{'hyb':>6}  {'rank':>7}  {'total':>7}"
    )
    print(header)
    print("-" * len(header))

    async with async_session_factory() as session:
        for i, question in enumerate(questions, start=1):
            result = await retrieve_context(
                session,
                embedder,
                reranker,
                tenant_id=DEMO_TENANT_ID,
                question=question,
                strategy=strategy,
            )
            top = result.context[0] if result.context else None
            where = (
                f"{top.filename} {top.section_path or '-'}"[:34] if top else "(no hits)"
            )
            t = result.timing
            print(
                f"{i:>2}  {question[:52]:<52}  {len(result.context):>3}  {where:<34}  "
                f"{t.hybrid_ms:>6.0f}  {t.rerank_ms:>7.0f}  {t.total_ms:>7.0f}"
            )

    print(
        f"\n{len(questions)} questions, strategy="
        f"{strategy or settings.retrieval_strategy}, reranker={reranker.name}"
    )


def main() -> int:
    sys.stdout.reconfigure(errors="replace", line_buffering=True)  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed_500", "structural"], default=None)
    parser.add_argument("--rerank", choices=["noop", "cross_encoder"], default="noop")
    args = parser.parse_args()
    asyncio.run(_run(args.strategy, args.rerank))
    return 0


if __name__ == "__main__":
    sys.exit(main())
