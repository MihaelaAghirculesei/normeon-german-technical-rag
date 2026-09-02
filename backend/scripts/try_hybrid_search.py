"""Ad-hoc check for Day 8: run German questions through the vector, the
full-text and the hybrid (RRF) retriever against the dev database and
print the top hits side by side.

Two things to look for in the HYBRID column:
- re-ranking: its top-3 differs from *both* single-branch top-3s, because
  a chunk ranked mid-list by one branch and well by the other is pulled
  up by consensus (e.g. a wrong vector #1 is displaced);
- promotion: a chunk the hybrid puts in its top-5 that sat outside the
  top-5 of *each* branch on its own -- the case for fusing rather than
  picking one branch. On this corpus FTS recall is thin (Day 7's
  AND-semantics limit), so re-ranking is the more common of the two.

Run with:
    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/try_hybrid_search.py
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
from app.services.retrieval import fts_search, hybrid_search, vector_search

# Two common terms each: `websearch_to_tsquery` ANDs them, so the FTS
# branch still returns a full ranking (a five-term question matches almost
# nothing -- the Day 7 AND-semantics limit). With both branches producing
# a real ranking that only partly agrees, RRF consensus can lift a chunk
# that sat mid-list in each branch above one that led only a single branch.
QUERIES = [
    "Scheinwerfer Fahrbahn",
    "Bremse Anhaenger",
    "Bereifung Achse",
    "Sitze Sicherheitsgurte",
    "Kennzeichen Beleuchtung",
]
_CANDIDATE_K = 40


def _ids(hits: list[RetrievedChunk]) -> list[str]:
    return [str(h.chunk_id) for h in hits]


def _fmt(hits: list[RetrievedChunk]) -> str:
    if not hits:
        return "      (no hits)"
    lines = []
    for rank, hit in enumerate(hits[:5], start=1):
        section = hit.section_path or "-"
        snippet = " ".join(hit.content.split())[:76]
        lines.append(
            f"      {rank}. {hit.score:.4f} {hit.filename} S.{hit.page_from} "
            f"[{section}] {snippet}"
        )
    return "\n".join(lines)


async def _compare(
    session: AsyncSession,
    embedder: EmbeddingAdapter,
    question: str,
    strategy: str | None,
    k: int,
) -> None:
    vec_hits = await vector_search(
        session, embedder, tenant_id=DEMO_TENANT_ID, question=question,
        strategy=strategy, k=k,
    )
    fts_hits = await fts_search(
        session, tenant_id=DEMO_TENANT_ID, question=question, strategy=strategy, k=k
    )
    hyb_hits = await hybrid_search(
        session, embedder, tenant_id=DEMO_TENANT_ID, question=question,
        strategy=strategy, candidate_k=_CANDIDATE_K, top_k=k,
    )

    def rank_in(hits: list[RetrievedChunk], chunk_id: str) -> str:
        ids = _ids(hits)
        return f"#{ids.index(chunk_id) + 1}" if chunk_id in ids else "-"

    top3_vec, top3_fts = _ids(vec_hits[:3]), _ids(fts_hits[:3])
    top3_hyb = _ids(hyb_hits[:3])
    reranked = top3_hyb != top3_vec and top3_hyb != top3_fts

    vec_top5, fts_top5 = set(_ids(vec_hits[:5])), set(_ids(fts_hits[:5]))
    promoted = [
        h for h in hyb_hits[:5] if str(h.chunk_id) not in vec_top5 | fts_top5
    ]

    print(f"\nQ: {question}")
    print("  VECTOR:")
    print(_fmt(vec_hits))
    print("  FTS:")
    print(_fmt(fts_hits))
    print("  HYBRID:")
    print(_fmt(hyb_hits))
    if reranked:
        print("  -> hybrid top-3 is a re-ranking distinct from both branch top-3s")
    if promoted:
        note = "; ".join(
            f"#{hyb_hits.index(h) + 1} {h.filename} S.{h.page_from} "
            f"[{h.section_path or '-'}] (vector {rank_in(vec_hits, str(h.chunk_id))}, "
            f"fts {rank_in(fts_hits, str(h.chunk_id))})"
            for h in promoted
        )
        print(f"  -> hybrid promoted into its top-5 from outside both branch top-5s: {note}")
    if not reranked and not promoted:
        print("  -> hybrid top-5 matches a single branch's ordering")


async def _run(strategy: str | None, k: int) -> None:
    embedder = get_embedder()
    await asyncio.to_thread(embedder.embed_query, "warmup")

    async with async_session_factory() as session:
        for question in QUERIES:
            await _compare(session, embedder, question, strategy, k)


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]  # Windows console code page
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed_500", "structural"], default=None)
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(_run(args.strategy, args.k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
