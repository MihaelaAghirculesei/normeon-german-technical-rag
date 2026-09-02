"""Ad-hoc check for Day 9: measure the cross-encoder reranker on the dev
corpus and show how it reorders a candidate set.

Candidates come from the full-text branch (`fts_search`), on purpose: it
needs no embedder, so this process only loads the reranker -- e5-large
and bge-reranker-v2-m3 together do not fit in RAM on the dev box.

For each question it pulls ~40 FTS candidates, times a single `rerank()`
call over all of them, and prints the top hits before and after. The
headline number is the per-call rerank latency on CPU (the plan expects
~1.5 s for 40 pairs) -- printed per question and as a mean.

Run with:
    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/try_rerank.py
"""

import argparse
import asyncio
import statistics
import sys
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.reranker.base import Reranker
from app.api.deps import get_reranker
from app.core.config import settings
from app.db.seed import DEMO_TENANT_ID
from app.db.session import async_session_factory
from app.domain.models import RetrievedChunk
from app.services.retrieval import fts_search

QUERIES = [
    "Scheinwerfer Fahrbahn",
    "Bremse Anhaenger",
    "Sitze Sicherheitsgurte",
    "Reifen Profiltiefe",
    "Kennzeichen Beleuchtung",
]


def _fmt(hits: list[RetrievedChunk], n: int = 3) -> str:
    if not hits:
        return "      (no hits)"
    return "\n".join(
        f"      {rank}. {hit.score:+.3f} {hit.filename} S.{hit.page_from} "
        f"[{hit.section_path or '-'}] {' '.join(hit.content.split())[:70]}"
        for rank, hit in enumerate(hits[:n], start=1)
    )


async def _one(session: AsyncSession, reranker: Reranker, question: str) -> float:
    candidates = await fts_search(
        session, tenant_id=DEMO_TENANT_ID, question=question, k=settings.hybrid_candidate_k
    )
    if not candidates:
        print(f"\nQ: {question}\n  (no FTS candidates -- skipped)")
        return 0.0

    started = time.perf_counter()
    reranked = await asyncio.to_thread(
        reranker.rerank, question, candidates, settings.rerank_top_k
    )
    elapsed = time.perf_counter() - started

    print(f"\nQ: {question}")
    print(
        f"  {len(candidates)} candidates -> rerank {elapsed * 1000:.0f} ms "
        f"({elapsed / len(candidates) * 1000:.0f} ms/pair)"
    )
    print("  FTS top-3:")
    print(_fmt(candidates))
    print("  RERANKED top-3:")
    print(_fmt(reranked))
    moved = candidates[0].chunk_id != reranked[0].chunk_id
    print(f"  -> reranker {'changed' if moved else 'kept'} the #1 chunk")
    return elapsed


async def _run() -> None:
    reranker = get_reranker()
    timings: list[float] = []
    async with async_session_factory() as session:
        for i, question in enumerate(QUERIES):
            elapsed = await _one(session, reranker, question)
            if i == 0:
                print("  (first call also pays the one-time model load)")
            elif elapsed > 0:
                timings.append(elapsed)

    if timings:
        print(
            f"\nrerank latency, questions 2..{len(QUERIES)} (model already loaded): "
            f"mean {statistics.mean(timings) * 1000:.0f} ms, "
            f"max {max(timings) * 1000:.0f} ms  (reranker={reranker.name})"
        )


def main() -> int:
    # line_buffering so progress is visible when stdout is redirected to a
    # file (the model is slow enough on CPU that a buffered run looks hung)
    sys.stdout.reconfigure(errors="replace", line_buffering=True)  # type: ignore[union-attr]
    argparse.ArgumentParser(description=__doc__).parse_args()
    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
