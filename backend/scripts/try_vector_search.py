"""Ad-hoc check for Day 6: run a handful of German questions through the
vector retriever against the dev database and print the top hits, with the
embedding time and the DB time reported separately.

Run with:
    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/try_vector_search.py

Acceptance bar (plan, Giorno 6): plausible chunks back from the DB in
<150 ms. The e5-large query embedding on CPU is measured on its own line
so it isn't counted against that bar.
"""

import argparse
import asyncio
import sys
import time

from app.api.deps import get_embedder
from app.db.seed import DEMO_TENANT_ID
from app.db.session import async_session_factory
from app.services.retrieval import vector_search_by_vector

QUESTIONS = [
    "Welche Anforderungen gelten fuer das amtliche Kennzeichen eines Fahrzeugs?",
    "Wann muss ein Fahrzeug zur Hauptuntersuchung vorgefuehrt werden?",
    "Welche Vorschriften gelten fuer Scheinwerfer und Beleuchtungseinrichtungen?",
    "Was regelt die Zulassung von Anhaengern im Strassenverkehr?",
    "Wann erlischt die Betriebserlaubnis eines umgebauten Fahrzeugs?",
]


async def _run(strategy: str | None, k: int) -> None:
    embedder = get_embedder()

    # Warm the model so its one-off load doesn't land on the first question.
    warmup_vector = await asyncio.to_thread(embedder.embed_query, "warmup")

    async with async_session_factory() as session:
        # And warm the HNSW index pages into shared_buffers, so the numbers
        # below reflect steady state rather than a cold first request.
        await vector_search_by_vector(
            session, tenant_id=DEMO_TENANT_ID, query_vector=warmup_vector,
            model=embedder.name, strategy=strategy, k=k,
        )

        for question in QUESTIONS:
            t0 = time.perf_counter()
            query_vector = await asyncio.to_thread(embedder.embed_query, question)
            t1 = time.perf_counter()
            hits = await vector_search_by_vector(
                session,
                tenant_id=DEMO_TENANT_ID,
                query_vector=query_vector,
                model=embedder.name,
                strategy=strategy,
                k=k,
            )
            t2 = time.perf_counter()

            embed_ms = (t1 - t0) * 1000
            search_ms = (t2 - t1) * 1000
            flag = "" if search_ms < 150 else "  <-- over 150 ms"
            print(f"\nQ: {question}")
            print(f"   embed {embed_ms:.0f} ms | search {search_ms:.0f} ms{flag}")
            for rank, hit in enumerate(hits, start=1):
                section = hit.section_path or "-"
                snippet = " ".join(hit.content.split())[:110]
                print(
                    f"   {rank}. {hit.score:.3f}  {hit.filename} S.{hit.page_from} "
                    f"[{section}]  {snippet}"
                )


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
