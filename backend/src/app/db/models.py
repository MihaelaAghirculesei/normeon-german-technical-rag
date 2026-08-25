import uuid
from datetime import date, datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Boolean, ForeignKey, Numeric, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    name: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(server_default=text("now()"))


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("tenant_id", "content_hash", name="uq_doc_tenant_hash"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"))
    content_hash: Mapped[str] = mapped_column(String(64))
    filename: Mapped[str]
    doc_type: Mapped[str]
    language: Mapped[str] = mapped_column(server_default="de")
    version_label: Mapped[str | None]
    valid_from: Mapped[date | None]
    valid_until: Mapped[date | None]
    page_count: Mapped[int | None]
    uploaded_by: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(server_default=text("now()"))


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "strategy", "chunk_hash", name="uq_chunk"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE")
    )
    tenant_id: Mapped[uuid.UUID]
    strategy: Mapped[str]
    chunk_hash: Mapped[str] = mapped_column(String(64))
    parent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chunks.id"))
    ordinal: Mapped[int]
    page_from: Mapped[int]
    page_to: Mapped[int]
    section_path: Mapped[str | None]
    heading: Mapped[str | None]
    char_start: Mapped[int]
    char_end: Mapped[int]
    content: Mapped[str]
    content_norm: Mapped[str]
    token_count: Mapped[int]


class Embedding(Base):
    __tablename__ = "embeddings"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(primary_key=True)
    dim: Mapped[int]
    vec: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    request_id: Mapped[str]
    tenant_id: Mapped[uuid.UUID]
    question: Mapped[str]
    config_hash: Mapped[str] = mapped_column(String(64))
    retrieved_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)))
    answer: Mapped[str | None]
    abstained: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    latency_ms: Mapped[dict[str, int]] = mapped_column(JSONB)
    tokens: Mapped[dict[str, int]] = mapped_column(JSONB)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    created_at: Mapped[datetime] = mapped_column(server_default=text("now()"))
