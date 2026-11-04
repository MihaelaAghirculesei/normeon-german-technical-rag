"""Fill chunks.regions for PDF chunks ingested before positions were kept,
and store the originals of the corpus PDFs on the way.

A chunk is matched by strategy and content hash, so chunk ids and embeddings
stay untouched. Idempotent: a chunk that has regions is left alone.

Run with:
    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/backfill_chunk_regions.py
"""

import asyncio
import hashlib
import sys
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.parsing.pdf import parse_pdf
from app.adapters.storage.local import default_storage
from app.db.models import Chunk, Document
from app.db.session import async_session_factory
from app.domain.chunking import ALL_STRATEGIES
from app.services.ingestion import region_rows, store_document_file

CORPUS_DIR = Path(__file__).parents[2] / "corpus"


async def _backfill(session: AsyncSession, document: Document, path: Path) -> int:
    parsed = await asyncio.to_thread(parse_pdf, path)
    filled = 0
    for strategy in ALL_STRATEGIES:
        for domain_chunk in strategy.chunk(parsed):
            chunk_hash = hashlib.sha256(domain_chunk.content.encode("utf-8")).hexdigest()
            result = await session.execute(
                update(Chunk)
                .where(
                    Chunk.document_id == document.id,
                    Chunk.strategy == strategy.name,
                    Chunk.chunk_hash == chunk_hash,
                    Chunk.regions.is_(None),
                )
                .values(regions=region_rows(domain_chunk))
            )
            assert isinstance(result, CursorResult)
            filled += result.rowcount
    await session.commit()
    return filled


async def _run() -> None:
    storage = default_storage()
    async with async_session_factory() as session:
        for path in sorted(CORPUS_DIR.glob("*.pdf")):
            content = path.read_bytes()
            content_hash = hashlib.sha256(content).hexdigest()
            documents = await session.scalars(
                select(Document).where(Document.content_hash == content_hash)
            )
            for document in documents.all():
                await store_document_file(storage, document, content)
                filled = await _backfill(session, document, path)
                print(f"{path.name} ({document.tenant_id}): {filled} chunks filled")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run())
