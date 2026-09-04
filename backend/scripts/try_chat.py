"""Ad-hoc check for Day 11: run a handful of German questions through the
full answer path (retrieve -> context -> prompt -> LLM) against the dev
database and print the answer plus the backend-held source table.

Run with (fake LLM, no key needed):
    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/try_chat.py

Point it at a real model by setting, in the environment or .env:
    LLM_PROVIDER=openai_compatible
    LLM_API_BASE_URL=https://<host>/v1
    LLM_API_KEY=<key>
    LLM_MODEL=<model-id>

Acceptance bar (plan, Giorno 11): a question yields a German answer with
`[S1]` markers, and every marker maps to a real page/section.
"""

import argparse
import asyncio
import sys

from app.api.deps import get_embedder, get_llm_client, get_reranker
from app.core.config import settings
from app.db.seed import DEMO_TENANT_ID
from app.db.session import async_session_factory
from app.services.generation import generate_answer

QUESTIONS = [
    "Welche Vorschriften gelten fuer Scheinwerfer und Beleuchtungseinrichtungen?",
    "Was ist ueber Sicherheitsgurte und ihre Verankerung geregelt?",
    "Welche Anforderungen bestehen an die Bereifung der Achsen?",
    "Wie ist die zulaessige Anhaengelast zu bestimmen?",
    "Wie hoch ist der Oelpreis in Katar?",  # expect NICHT_GEFUNDEN
]


async def _run(strategy: str | None) -> None:
    embedder = get_embedder()
    reranker = get_reranker()
    llm = get_llm_client()

    print(f"llm_provider={settings.llm_provider} model={settings.llm_model or '(n/a)'}\n")

    async with async_session_factory() as session:
        for question in QUESTIONS:
            result = await generate_answer(
                session, embedder, reranker, llm,
                tenant_id=DEMO_TENANT_ID, question=question, strategy=strategy,
            )
            t = result.retrieval_timing
            print(f"Q: {question}")
            print(
                f"   retrieval {t.total_ms:.0f} ms "
                f"(hybrid {t.hybrid_ms:.0f} / rerank {t.rerank_ms:.0f} / "
                f"select {t.select_ms:.0f}) | generation {result.generation_ms:.0f} ms"
            )
            print(f"   prompt {result.prompt_name} ({result.prompt_sha256[:12]}) "
                  f"| model {result.model}")
            print(f"   A: {result.answer}")
            for src in result.sources:
                section = src.section_path or "-"
                page = (
                    f"S.{src.page_from}"
                    if src.page_to == src.page_from
                    else f"S.{src.page_from}-{src.page_to}"
                )
                print(f"      [{src.marker}] {src.filename} {page} [{section}]")
            print()


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]  # Windows console code page
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed_500", "structural"], default=None)
    args = parser.parse_args()
    asyncio.run(_run(args.strategy))
    return 0


if __name__ == "__main__":
    sys.exit(main())
