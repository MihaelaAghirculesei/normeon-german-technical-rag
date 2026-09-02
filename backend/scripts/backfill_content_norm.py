"""Recompute chunks.content_norm with the real normalize_de.

Day 5 ingested the corpus with a `content.lower()` placeholder for
content_norm; Day 7 replaced it with normalize_de. This rewrites the
column for every existing chunk -- the `tsv` generated column and its GIN
index follow automatically on update. Idempotent (normalize_de is), so
re-running is a no-op.

Run with:
    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/backfill_content_norm.py
"""

import asyncio
import sys
import time

from sqlalchemy import func, select, update

from app.db.models import Chunk
from app.db.session import async_session_factory
from app.domain.normalization import normalize_de

BATCH = 500


async def _run() -> None:
    async with async_session_factory() as session:
        total = await session.scalar(select(func.count()).select_from(Chunk))
        print(f"{total} chunks to check")

        changed = 0
        last_id = None
        started = time.monotonic()
        while True:
            query = select(Chunk.id, Chunk.content, Chunk.content_norm).order_by(Chunk.id).limit(
                BATCH
            )
            if last_id is not None:
                query = query.where(Chunk.id > last_id)
            rows = (await session.execute(query)).all()
            if not rows:
                break
            for chunk_id, content, current_norm in rows:
                new_norm = normalize_de(content)
                if new_norm != current_norm:
                    await session.execute(
                        update(Chunk).where(Chunk.id == chunk_id).values(content_norm=new_norm)
                    )
                    changed += 1
                last_id = chunk_id
            await session.commit()
            print(f"  ...through {last_id}  ({changed} rewritten)")

        elapsed = time.monotonic() - started
        print(f"done: {changed}/{total} rewritten in {elapsed:.0f}s")


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]  # Windows console code page
    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
