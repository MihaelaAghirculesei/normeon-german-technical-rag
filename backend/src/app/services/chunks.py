"""A cited chunk with the text around it (Giorno 23): what the citation
panel shows when a reader clicks `[S1]` -- the chunk itself, highlighted,
between its neighbours, so a sentence cut off at a chunk boundary can
still be read in full.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ChunkNotFoundError
from app.db.models import Chunk, Document


@dataclass(frozen=True, slots=True)
class ChunkContext:
    document: Document
    chunk: Chunk
    before: list[Chunk]
    after: list[Chunk]


async def get_chunk_context(
    session: AsyncSession, *, tenant_id: UUID, chunk_id: UUID, window: int
) -> ChunkContext:
    """The chunk plus up to `window` neighbours on each side, taken from
    the same document *and* chunking strategy (the two strategies cut the
    same text differently, so mixing them would repeat it) and ordered by
    `ordinal`. Raises `ChunkNotFoundError` for an unknown id or another
    tenant's chunk alike."""
    chunk = await session.scalar(
        select(Chunk).where(Chunk.id == chunk_id, Chunk.tenant_id == tenant_id)
    )
    if chunk is None:
        raise ChunkNotFoundError(f"chunk {chunk_id} not found")

    document = await session.get(Document, chunk.document_id)
    if document is None:  # FK + cascade make this unreachable in practice
        raise ChunkNotFoundError(f"chunk {chunk_id} not found")

    neighbours = (
        await session.scalars(
            select(Chunk)
            .where(
                Chunk.document_id == chunk.document_id,
                Chunk.strategy == chunk.strategy,
                Chunk.ordinal.between(chunk.ordinal - window, chunk.ordinal + window),
                Chunk.id != chunk.id,
            )
            .order_by(Chunk.ordinal)
        )
    ).all()
    return ChunkContext(
        document=document,
        chunk=chunk,
        before=[c for c in neighbours if c.ordinal < chunk.ordinal],
        after=[c for c in neighbours if c.ordinal > chunk.ordinal],
    )
