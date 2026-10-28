"""Embed the chunks that have no vector yet for the configured embedder.

Moving the corpus to another embedding model (Giorno 29: e5 -> Gemini)
means embedding every chunk again. Vectors are keyed by the embedder's
name, so the old model's rows stay and both can be compared. A batch at a
time, committed, with a pause between batches to stay inside a free
tier's tokens-per-minute; re-running resumes where it stopped.

Run with:
    cd backend
    DATABASE_URL=... EMBEDDING_PROVIDER=gemini EMBEDDING_MODEL=gemini-embedding-2 \\
    EMBEDDING_API_KEY=... .venv/Scripts/python scripts/embed_missing.py --strategy structural
"""

import argparse
import asyncio
import sys
import time

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.api.deps import get_embedder
from app.db.models import Chunk, Embedding
from app.db.session import async_session_factory


async def _run(strategy: str | None, batch_size: int, interval_s: float) -> None:
    embedder = get_embedder()
    missing = (
        select(Chunk.id, Chunk.content)
        .outerjoin(Embedding, (Embedding.chunk_id == Chunk.id) & (Embedding.model == embedder.name))
        .where(Embedding.chunk_id.is_(None))
        .order_by(Chunk.id)
    )
    if strategy is not None:
        missing = missing.where(Chunk.strategy == strategy)

    async with async_session_factory() as session:
        total = await session.scalar(select(func.count()).select_from(missing.subquery()))
        print(f"{total} chunks without a {embedder.name} vector", flush=True)
        done = 0
        started = time.monotonic()
        while True:
            batch = (await session.execute(missing.limit(batch_size))).all()
            if not batch:
                break
            texts = [row.content for row in batch]
            vectors = await asyncio.to_thread(embedder.embed_passages, texts)
            await session.execute(
                pg_insert(Embedding)
                .values(
                    [
                        {"chunk_id": row.id, "model": embedder.name, "dim": embedder.dim, "vec": v}
                        for row, v in zip(batch, vectors, strict=True)
                    ]
                )
                .on_conflict_do_nothing(index_elements=["chunk_id", "model"])
            )
            await session.commit()
            done += len(batch)
            print(f"{done}/{total} ({time.monotonic() - started:.0f}s)", flush=True)
            if done < (total or 0):
                await asyncio.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--strategy", choices=["structural", "fixed_500"], default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between batches")
    args = parser.parse_args()
    asyncio.run(_run(args.strategy, args.batch_size, args.interval))


if __name__ == "__main__":
    sys.exit(main())
