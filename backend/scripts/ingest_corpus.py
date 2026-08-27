"""Ingest the demo corpus (the two real German traffic-law PDFs) into the
demo tenant.

Run with:
    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/ingest_corpus.py

Idempotent: re-running skips a document already ingested (same tenant and
content hash) and resumes one left mid-pipeline. On first run the local
e5-large model is downloaded and every chunk is embedded on CPU, so this
takes a while.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from sqlalchemy import func, select

from app.api.deps import get_embedder
from app.db.models import Chunk, Tenant
from app.db.seed import DEMO_TENANT_ID
from app.db.session import async_session_factory
from app.services.ingestion import ingest_document

CORPUS_DIR = Path(__file__).parents[2] / "corpus"
DEFAULT_FILES = ("StVZO.pdf", "FZV.pdf")


async def _ensure_demo_tenant() -> None:
    async with async_session_factory() as session:
        existing = await session.scalar(select(Tenant).where(Tenant.id == DEMO_TENANT_ID))
        if existing is None:
            session.add(Tenant(id=DEMO_TENANT_ID, name="Demo Tenant"))
            await session.commit()
            print(f"seeded demo tenant {DEMO_TENANT_ID}")


async def _ingest(files: list[str]) -> None:
    embedder = get_embedder()
    for name in files:
        path = CORPUS_DIR / name
        if not path.is_file():
            print(f"{name}: SKIPPED (not found at {path})")
            continue
        content = path.read_bytes()
        started = time.monotonic()
        async with async_session_factory() as session:
            document, already_ingested = await ingest_document(
                session, embedder, DEMO_TENANT_ID, name, "norm", content
            )
            chunk_count = await session.scalar(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.document_id == document.id)
            )
        outcome = "already ingested" if already_ingested else f"ingested ({document.status})"
        elapsed = time.monotonic() - started
        print(f"{name}: {outcome}, {chunk_count} chunks, {elapsed:.0f}s  document_id={document.id}")


async def _run(files: list[str]) -> None:
    await _ensure_demo_tenant()
    await _ingest(files)


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]  # Windows console code page
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", default=list(DEFAULT_FILES))
    args = parser.parse_args()
    asyncio.run(_run(args.files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
