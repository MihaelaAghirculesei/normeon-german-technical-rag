"""GET /api/v1/chunks/{chunk_id}/context (Giorno 23): the chunk behind a
citation, with its neighbours, for the citation panel."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.api.deps import AuthDep, TenantSession
from app.db.models import Chunk
from app.services.chunks import get_chunk_context

router = APIRouter(prefix="/api/v1/chunks", tags=["chunks"])


class ChunkOut(BaseModel):
    chunk_id: UUID
    ordinal: int
    page_from: int
    page_to: int
    section_path: str | None
    heading: str | None
    content: str
    ocr: bool  # recognised by OCR from a scanned page


class ContextDocumentOut(BaseModel):
    document_id: UUID
    filename: str
    doc_type: str
    version_label: str | None


class ChunkContextResponse(BaseModel):
    document: ContextDocumentOut
    chunk: ChunkOut
    before: list[ChunkOut]
    after: list[ChunkOut]


def _chunk_out(chunk: Chunk) -> ChunkOut:
    return ChunkOut(
        chunk_id=chunk.id,
        ordinal=chunk.ordinal,
        page_from=chunk.page_from,
        page_to=chunk.page_to,
        section_path=chunk.section_path,
        heading=chunk.heading,
        content=chunk.content,
        ocr=chunk.ocr,
    )


@router.get("/{chunk_id}/context", response_model=ChunkContextResponse)
async def chunk_context(
    chunk_id: UUID,
    auth: AuthDep,
    session: TenantSession,
    window: Annotated[int, Query(ge=0, le=3)] = 1,
) -> ChunkContextResponse:
    context = await get_chunk_context(
        session, tenant_id=auth.tenant_id, chunk_id=chunk_id, window=window
    )
    return ChunkContextResponse(
        document=ContextDocumentOut(
            document_id=context.document.id,
            filename=context.document.filename,
            doc_type=context.document.doc_type,
            version_label=context.document.version_label,
        ),
        chunk=_chunk_out(context.chunk),
        before=[_chunk_out(c) for c in context.before],
        after=[_chunk_out(c) for c in context.after],
    )
