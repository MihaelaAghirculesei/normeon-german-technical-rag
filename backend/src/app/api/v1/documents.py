from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, Form, Response, UploadFile
from pydantic import BaseModel

from app.adapters.embedding.base import EmbeddingAdapter
from app.api.deps import DbSession, EmbedderDep
from app.core.errors import DocumentNotFoundError
from app.db.models import Document
from app.db.session import async_session_factory
from app.services.ingestion import get_or_create_document, process_document

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


class DocumentStatusResponse(BaseModel):
    document_id: UUID
    filename: str
    status: str
    already_ingested: bool = False


@router.post("", response_model=DocumentStatusResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    session: DbSession,
    embedder: EmbedderDep,
    response: Response,
    tenant_id: Annotated[UUID, Form()],
    doc_type: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
) -> DocumentStatusResponse:
    content = await file.read()
    document, already_ingested = await get_or_create_document(
        session, tenant_id, file.filename or "unnamed", doc_type, content
    )
    if already_ingested:
        # Nothing was created, so 201 would be a lie -- report the existing
        # resource with a plain 200 instead.
        response.status_code = 200
    else:
        background_tasks.add_task(_process_in_background, document.id, content, embedder)
    return DocumentStatusResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status,
        already_ingested=already_ingested,
    )


async def _process_in_background(
    document_id: UUID, content: bytes, embedder: EmbeddingAdapter
) -> None:
    # Runs after the response has been sent, so it gets its own session
    # rather than reusing the request's (already torn down by then).
    async with async_session_factory() as session:
        document = await session.get(Document, document_id)
        if document is not None:
            await process_document(session, embedder, document, content)


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(document_id: UUID, session: DbSession) -> DocumentStatusResponse:
    document = await session.get(Document, document_id)
    if document is None:
        raise DocumentNotFoundError(f"document {document_id} not found")
    return DocumentStatusResponse(
        document_id=document.id, filename=document.filename, status=document.status
    )
