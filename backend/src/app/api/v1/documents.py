import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, Form, Response, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select

from app.adapters.embedding.base import EmbeddingAdapter
from app.adapters.parsing.inspect import UploadLimits, inspect_upload
from app.api.deps import AdminDep, AuthDep, EmbedderDep, TenantSession
from app.core.config import settings
from app.core.errors import (
    DocumentNotFoundError,
    MalformedDocumentError,
    UnsupportedFileTypeError,
)
from app.db.models import Chunk, Document
from app.db.session import tenant_session
from app.services.ingestion import SUPPORTED_SUFFIXES, get_or_create_document, process_document

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


class DocumentStatusResponse(BaseModel):
    document_id: UUID
    filename: str
    status: str
    # Why ingestion failed: parse_timeout, parse_failed or embedding_failed;
    # null unless status is "failed".
    error_code: str | None


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
    auth: AdminDep,
    session: TenantSession,
    embedder: EmbedderDep,
    response: Response,
    doc_type: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    version_label: Annotated[str | None, Form()] = None,
    valid_from: Annotated[date | None, Form()] = None,
    valid_until: Annotated[date | None, Form()] = None,
) -> DocumentUploadResponse:
    filename = file.filename or "unnamed"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        # Rejected before anything is stored: a Document row for a file no
        # parser can read would only ever end up "failed".
        raise UnsupportedFileTypeError(
            f"{suffix or 'no extension'} is not supported; upload "
            + " or ".join(sorted(SUPPORTED_SUFFIXES))
        )

    content = await file.read()
    # Opened and checked before anything is stored: an encrypted, broken or
    # zip-bomb file is refused here with a clear error instead of becoming
    # a document stuck in "failed" (Giorno 26).
    try:
        await asyncio.wait_for(
            asyncio.to_thread(inspect_upload, suffix, content, _upload_limits()),
            settings.upload_check_timeout_s,
        )
    except TimeoutError as exc:
        raise MalformedDocumentError("the file could not be checked in time") from exc

    document, created = await get_or_create_document(
        session,
        auth.tenant_id,
        filename,
        doc_type,
        content,
        version_label=version_label or None,
        valid_from=valid_from,
        valid_until=valid_until,
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
        background_tasks.add_task(
            _process_in_background, auth.tenant_id, document.id, content, embedder
        )
    if not created:
        # Nothing was created, so 201 would be a lie -- report the existing
        # resource with a plain 200 instead.
        response.status_code = 200
    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status,
        error_code=document.error_code,
        outcome=outcome,
    )


def _upload_limits() -> UploadLimits:
    return UploadLimits(
        max_pages=settings.max_document_pages,
        max_uncompressed_bytes=settings.max_docx_uncompressed_mb * 2**20,
        max_compression_ratio=settings.max_docx_compression_ratio,
    )


async def _process_in_background(
    tenant_id: UUID, document_id: UUID, content: bytes, embedder: EmbeddingAdapter
) -> None:
    # Runs after the response has been sent, so it gets its own session
    # rather than reusing the request's (already torn down by then) --
    # scoped to the same tenant, so ingestion writes under RLS as well.
    async with tenant_session(tenant_id) as session:
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
    error_code: str | None
    created_at: datetime
    # Chunks per chunking strategy, e.g. {"fixed_500": 233, "structural": 114};
    # empty until the document has been parsed.
    chunk_counts: dict[str, int]


@router.get("", response_model=list[DocumentSummary])
async def list_documents(auth: AuthDep, session: TenantSession) -> list[DocumentSummary]:
    tenant_id = auth.tenant_id
    documents = await session.scalars(
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .order_by(Document.filename, Document.version_label)
    )
    counted = await session.execute(
        select(Chunk.document_id, Chunk.strategy, func.count())
        .where(Chunk.tenant_id == tenant_id)
        .group_by(Chunk.document_id, Chunk.strategy)
    )
    chunk_counts: dict[UUID, dict[str, int]] = {}
    for document_id, strategy, count in counted:
        chunk_counts.setdefault(document_id, {})[strategy] = count
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
            error_code=d.error_code,
            created_at=d.created_at,
            chunk_counts=chunk_counts.get(d.id, {}),
        )
        for d in documents
    ]


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: UUID, auth: AuthDep, session: TenantSession
) -> DocumentStatusResponse:
    # Another tenant's document is reported exactly like a missing one.
    document = await session.scalar(
        select(Document).where(Document.id == document_id, Document.tenant_id == auth.tenant_id)
    )
    if document is None:
        raise DocumentNotFoundError(f"document {document_id} not found")
    return DocumentStatusResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status,
        error_code=document.error_code,
    )
