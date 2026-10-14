from datetime import date, datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, Form, Query, Response, UploadFile
from pydantic import BaseModel
from sqlalchemy import select

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


UploadOutcome = Literal["created", "duplicate", "in_progress", "retrying"]


class DocumentUploadResponse(DocumentStatusResponse):
    # Only meaningful as the answer to an upload, so not on the status read:
    #   created     -- new document, processing scheduled (201)
    #   duplicate   -- identical file already ingested, nothing to do (200)
    #   in_progress -- identical file is being processed right now; it is
    #                  not scheduled a second time (200)
    #   retrying    -- identical file failed before; processing resumes (200)
    outcome: UploadOutcome


_IN_FLIGHT = frozenset({"pending", "parsing", "embedding"})


@router.post("", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    session: DbSession,
    embedder: EmbedderDep,
    response: Response,
    tenant_id: Annotated[UUID, Form()],
    doc_type: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
) -> DocumentUploadResponse:
    content = await file.read()
    document, created = await get_or_create_document(
        session, tenant_id, file.filename or "unnamed", doc_type, content
    )

    outcome: UploadOutcome
    if created:
        outcome = "created"
    elif document.status == "ready":
        outcome = "duplicate"
    elif document.status in _IN_FLIGHT:
        # Known limit: a document left in flight by a crashed process stays
        # so until scripts/ingest_corpus.py (or a later upload after it is
        # marked failed) resumes it.
        outcome = "in_progress"
    else:
        outcome = "retrying"

    if outcome in ("created", "retrying"):
        background_tasks.add_task(_process_in_background, document.id, content, embedder)
    if not created:
        # Nothing was created, so 201 would be a lie -- report the existing
        # resource with a plain 200 instead.
        response.status_code = 200
    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status,
        outcome=outcome,
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


class DocumentSummary(BaseModel):
    document_id: UUID
    filename: str
    doc_type: str
    version_label: str | None
    valid_from: date | None
    valid_until: date | None
    page_count: int | None
    status: str
    created_at: datetime


@router.get("", response_model=list[DocumentSummary])
async def list_documents(
    session: DbSession, tenant_id: Annotated[UUID, Query()]
) -> list[DocumentSummary]:
    # tenant_id is an explicit parameter until auth lands (Giorno 25);
    # then it comes from the token instead.
    documents = await session.scalars(
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .order_by(Document.filename, Document.version_label)
    )
    return [
        DocumentSummary(
            document_id=d.id,
            filename=d.filename,
            doc_type=d.doc_type,
            version_label=d.version_label,
            valid_from=d.valid_from,
            valid_until=d.valid_until,
            page_count=d.page_count,
            status=d.status,
            created_at=d.created_at,
        )
        for d in documents
    ]


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(document_id: UUID, session: DbSession) -> DocumentStatusResponse:
    document = await session.get(Document, document_id)
    if document is None:
        raise DocumentNotFoundError(f"document {document_id} not found")
    return DocumentStatusResponse(
        document_id=document.id, filename=document.filename, status=document.status
    )
