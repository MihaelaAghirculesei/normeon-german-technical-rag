"""Ad-hoc check for Day 7: run a few German questions through both the
full-text and the vector retriever against the dev database and print the
top hits side by side. The point is the code-shaped queries -- where FTS
lands the exact section and the pure-vector branch drifts -- which is the
argument for the hybrid retriever coming on Day 8.

Run with:
    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/try_fts_search.py
"""

import argparse
import asyncio
import sys

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embedding.base import EmbeddingAdapter
from app.api.deps import get_embedder
from app.db.seed import DEMO_TENANT_ID
from app.db.session import async_session_factory
from app.domain.models import RetrievedChunk
from app.services.retrieval import fts_search, vector_search

# Two kinds of query on purpose. The first group is where FTS is meant to
# win: exact terms and code references that appear verbatim in the text.
# The second is where the German stemmer's lack of compound splitting and
# tsquery's AND semantics hurt it -- and the vector branch carries it.
# That contrast is the case for the hybrid retriever on Day 8.
KEYWORD_QUERIES = [
    "ECE-R 83",
    "Abblendlicht Scheinwerfer Fahrbahn",
    "Sicherheitsgurte Rueckhaltesysteme",
]
PARAPHRASE_QUERIES = [
    "Wann muss ein Auto zum TUEV?",
    "Wie hell muessen die Frontleuchten strahlen?",
]


def _fmt(hits: list[RetrievedChunk]) -> str:
    if not hits:
        return "      (no hits)"
    lines = []
    for rank, hit in enumerate(hits[:3], start=1):
        section = hit.section_path or "-"
        snippet = " ".join(hit.content.split())[:80]
        lines.append(
            f"      {rank}. {hit.score:.3f} {hit.filename} S.{hit.page_from} [{section}] {snippet}"
        )
    return "\n".join(lines)


async def _compare(
    session: AsyncSession,
    embedder: EmbeddingAdapter,
    question: str,
    strategy: str | None,
    k: int,
) -> None:
    fts_hits = await fts_search(
        session, tenant_id=DEMO_TENANT_ID, question=question, strategy=strategy, k=k
    )
    vec_hits = await vector_search(
        session,
        embedder,
        tenant_id=DEMO_TENANT_ID,
        question=question,
        strategy=strategy,
        k=k,
    )
    print(f"\nQ: {question}")
    print("  FTS:")
    print(_fmt(fts_hits))
    print("  VECTOR:")
    print(_fmt(vec_hits))


async def _run(strategy: str | None, k: int) -> None:
    embedder = get_embedder()
    await asyncio.to_thread(embedder.embed_query, "warmup")

    async with async_session_factory() as session:
        print("=== keyword / code queries (FTS should be competitive) ===")
        for question in KEYWORD_QUERIES:
            await _compare(session, embedder, question, strategy, k)
        print("\n=== paraphrased queries (FTS weak: compounds + AND semantics) ===")
        for question in PARAPHRASE_QUERIES:
            await _compare(session, embedder, question, strategy, k)


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]  # Windows console code page
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed_500", "structural"], default=None)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(_run(args.strategy, args.k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
