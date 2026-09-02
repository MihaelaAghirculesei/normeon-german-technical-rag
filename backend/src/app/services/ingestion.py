import asyncio
import hashlib
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embedding.base import EmbeddingAdapter
from app.adapters.parsing.pdf import parse_pdf
from app.db.models import Chunk, Document, Embedding
from app.domain.chunking import ALL_STRATEGIES
from app.domain.chunking import Chunk as DomainChunk
from app.domain.normalization import normalize_de

EMBEDDING_BATCH_SIZE = 32


async def get_or_create_document(
    session: AsyncSession,
    tenant_id: UUID,
    filename: str,
    doc_type: str,
    content: bytes,
) -> tuple[Document, bool]:
    """Fast, synchronous half of ingestion: hash the upload and look up (or
    create) its Document row. Returns (document, already_ingested) so a
    caller (the upload endpoint) can respond immediately and only schedule
    the slow parse/chunk/embed work (`process_document`) when there's
    actually something left to do."""
    content_hash = hashlib.sha256(content).hexdigest()

    existing = await session.scalar(
        select(Document).where(
            Document.tenant_id == tenant_id, Document.content_hash == content_hash
        )
    )
    if existing is not None:
        return existing, existing.status == "ready"

    document = Document(
        tenant_id=tenant_id,
        content_hash=content_hash,
        filename=filename,
        doc_type=doc_type,
        status="pending",
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document, False


async def process_document(
    session: AsyncSession, embedder: EmbeddingAdapter, document: Document, content: bytes
) -> None:
    """The slow half: parse -> chunk (both strategies) -> embed every chunk
    that has no embedding yet for this model. Safe to call repeatedly on the
    same document: a `ready` document is a no-op, and one stuck mid-pipeline
    (pending/parsing/embedding/failed) resumes rather than restarting, since
    chunk_hash / (chunk_id, model) uniqueness makes re-inserting an
    already-inserted chunk or embedding a no-op, and embedding is driven off
    "chunks still missing an embedding" rather than "chunks inserted this
    run" -- so a document interrupted partway through embedding finishes its
    remaining chunks instead of being marked ready with gaps.

    Status transitions each commit on their own, rather than the whole run
    being one transaction, so GET /documents/{id} can observe live progress
    while a large document is still being embedded."""
    if document.status == "ready":
        return

    try:
        document.status = "parsing"
        await session.commit()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            parsed = await asyncio.to_thread(parse_pdf, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        document.page_count = max((b.page for b in parsed.blocks), default=0)

        for strategy in ALL_STRATEGIES:
            await _insert_chunks(session, document, strategy.name, strategy.chunk(parsed))

        document.status = "embedding"
        await session.commit()

        pending = await _chunks_missing_embeddings(session, document, embedder.name)
        await _embed_and_store(session, embedder, pending)

        document.status = "ready"
        await session.commit()
    except Exception:
        document.status = "failed"
        await session.commit()
        raise


async def ingest_document(
    session: AsyncSession,
    embedder: EmbeddingAdapter,
    tenant_id: UUID,
    filename: str,
    doc_type: str,
    content: bytes,
) -> tuple[Document, bool]:
    """Convenience wrapper running both halves in sequence -- used by the
    corpus-loading script and tests, which don't need the HTTP split between
    an immediate response and background processing."""
    document, already_ingested = await get_or_create_document(
        session, tenant_id, filename, doc_type, content
    )
    if already_ingested:
        return document, True
    await process_document(session, embedder, document, content)
    return document, False


def _chunk_row(document: Document, strategy_name: str, dc: DomainChunk) -> dict[str, Any]:
    return {
        "document_id": document.id,
        "tenant_id": document.tenant_id,
        "strategy": strategy_name,
        "chunk_hash": hashlib.sha256(dc.content.encode("utf-8")).hexdigest(),
        "ordinal": dc.ordinal,
        "page_from": dc.page_from,
        "page_to": dc.page_to,
        "section_path": dc.section_path,
        "heading": dc.heading,
        # Placeholder span (the chunk's own content, start to end) -- a real
        # offset into the source document isn't tracked yet; revisit if a
        # future day needs to highlight the original PDF text.
        "char_start": 0,
        "char_end": len(dc.content),
        "content": dc.content,
        # Feeds the `tsv` generated column (to_tsvector('german', ...)).
        # Same function normalises the query side, so the two agree.
        "content_norm": normalize_de(dc.content),
        "token_count": dc.token_count,
    }


async def _insert_chunks(
    session: AsyncSession,
    document: Document,
    strategy_name: str,
    domain_chunks: list[DomainChunk],
) -> None:
    if not domain_chunks:
        return

    rows = [_chunk_row(document, strategy_name, dc) for dc in domain_chunks]
    stmt = pg_insert(Chunk).values(rows).on_conflict_do_nothing(
        index_elements=["document_id", "strategy", "chunk_hash"]
    )
    await session.execute(stmt)

    # Resolve parent_ordinal -> parent_id now that every chunk of this
    # strategy has a real id -- including ones inserted in an earlier,
    # resumed attempt, not just the ones just inserted above.
    all_rows = await session.scalars(
        select(Chunk).where(Chunk.document_id == document.id, Chunk.strategy == strategy_name)
    )
    id_by_ordinal = {c.ordinal: c.id for c in all_rows}
    dc_by_ordinal = {dc.ordinal: dc for dc in domain_chunks}
    for ordinal, chunk_id in id_by_ordinal.items():
        dc = dc_by_ordinal.get(ordinal)
        if dc is None or dc.parent_ordinal is None:
            continue
        parent_id = id_by_ordinal.get(dc.parent_ordinal)
        if parent_id is not None:
            await session.execute(
                update(Chunk).where(Chunk.id == chunk_id).values(parent_id=parent_id)
            )
    await session.commit()


async def _chunks_missing_embeddings(
    session: AsyncSession, document: Document, model: str
) -> list[Chunk]:
    """Every chunk of this document without an embedding row for `model`,
    ordered so batching is deterministic. On a fresh document this is all of
    its chunks; on a resumed one it is only the chunks not yet embedded."""
    result = await session.scalars(
        select(Chunk)
        .outerjoin(Embedding, (Embedding.chunk_id == Chunk.id) & (Embedding.model == model))
        .where(Chunk.document_id == document.id, Embedding.chunk_id.is_(None))
        .order_by(Chunk.strategy, Chunk.ordinal)
    )
    return list(result)


async def _embed_and_store(
    session: AsyncSession, embedder: EmbeddingAdapter, chunks: list[Chunk]
) -> None:
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        vectors = await asyncio.to_thread(embedder.embed_passages, [c.content for c in batch])
        stmt = (
            pg_insert(Embedding)
            .values(
                [
                    {
                        "chunk_id": chunk.id,
                        "model": embedder.name,
                        "dim": embedder.dim,
                        "vec": vector,
                    }
                    for chunk, vector in zip(batch, vectors, strict=True)
                ]
            )
            .on_conflict_do_nothing(index_elements=["chunk_id", "model"])
        )
        await session.execute(stmt)
        await session.commit()
