"""Ingests the same PDF twice against a real Postgres and asserts the chunk
count doesn't change the second time -- the literal Day 5 acceptance
criterion from the plan, made into a regression test.
"""

import asyncio
from pathlib import Path

import pymupdf
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from app.db.models import Chunk, Document, Embedding, Tenant
from app.services.ingestion import ingest_document

MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"
EMBEDDING_DIM = 1024


class _FakeEmbedder:
    """Deterministic, near-instant stand-in for the real e5 model -- the
    ingestion pipeline only cares that something implements the Protocol."""

    name = "fake"
    dim = EMBEDDING_DIM

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dim


def _alembic_config(database_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


def _build_test_pdf(tmp_path: Path) -> bytes:
    """A small but structurally realistic PDF: a running header (repeated at
    the same spot on every page, so the parser filters it) plus three
    numbered sections whose headings and body sit at a different vertical
    offset on each page. Identical per-page geometry would make the
    repeated-header/footer filter treat the body as boilerplate and drop
    every block, leaving nothing to chunk."""
    sections = [
        ("1 Allgemeines", "Dieser Abschnitt regelt den Geltungsbereich der Verordnung. " * 20),
        ("2 Begriffsbestimmungen", "Im Sinne dieser Verordnung gelten folgende Begriffe. " * 20),
        ("3 Technische Anforderungen", "Fahrzeuge muessen den Anforderungen entsprechen. " * 20),
    ]
    doc = pymupdf.open()
    for page_no, (title, body) in enumerate(sections, start=1):
        page = doc.new_page()
        offset = page_no * 18
        page.insert_text((50, 40), "Technische Norm - Entwurf", fontsize=8, fontname="helv")
        page.insert_text((50, 90 + offset), title, fontsize=15, fontname="hebo")
        page.insert_textbox((50, 120 + offset, 545, 790), body, fontsize=11, fontname="helv")
    path = tmp_path / "test.pdf"
    doc.save(path)
    doc.close()
    return path.read_bytes()


def test_ingesting_the_same_pdf_twice_does_not_change_chunk_count(tmp_path: Path) -> None:
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        database_url = pg.get_connection_url()
        command.upgrade(_alembic_config(database_url), "head")
        asyncio.run(_ingest_twice_and_assert(database_url, tmp_path))


async def _ingest_twice_and_assert(database_url: str, tmp_path: Path) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    embedder = _FakeEmbedder()
    content = _build_test_pdf(tmp_path)

    async with session_factory() as session:
        tenant = Tenant(name="Demo Tenant")
        session.add(tenant)
        await session.flush()
        tenant_id = tenant.id
        await session.commit()

    async with session_factory() as session:
        document_first, already_ingested_first = await ingest_document(
            session, embedder, tenant_id, "test.pdf", "norm", content
        )
        assert already_ingested_first is False
        assert document_first.status == "ready"

    async with session_factory() as session:
        chunk_count_after_first = await session.scalar(select(func.count()).select_from(Chunk))
        embedding_count_after_first = await session.scalar(
            select(func.count()).select_from(Embedding)
        )
        assert chunk_count_after_first is not None and chunk_count_after_first > 0
        assert embedding_count_after_first == chunk_count_after_first

    async with session_factory() as session:
        document_second, already_ingested_second = await ingest_document(
            session, embedder, tenant_id, "test.pdf", "norm", content
        )
        assert already_ingested_second is True
        assert document_second.id == document_first.id

    async with session_factory() as session:
        chunk_count_after_second = await session.scalar(select(func.count()).select_from(Chunk))
        assert chunk_count_after_second == chunk_count_after_first

    await engine.dispose()


def test_ingestion_produces_chunks_for_both_strategies(tmp_path: Path) -> None:
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        database_url = pg.get_connection_url()
        command.upgrade(_alembic_config(database_url), "head")
        asyncio.run(_ingest_once_and_check_strategies(database_url, tmp_path))


async def _ingest_once_and_check_strategies(database_url: str, tmp_path: Path) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    embedder = _FakeEmbedder()
    content = _build_test_pdf(tmp_path)

    async with session_factory() as session:
        tenant = Tenant(name="Demo Tenant")
        session.add(tenant)
        await session.commit()
        tenant_id = tenant.id

    async with session_factory() as session:
        await ingest_document(session, embedder, tenant_id, "test.pdf", "norm", content)

    async with session_factory() as session:
        strategies = await session.scalars(select(Chunk.strategy).distinct())
        assert set(strategies) == {"fixed_500", "structural"}

    await engine.dispose()


class _EmbedderFailingOnSecondBatch:
    """Succeeds on the first batch, then raises -- simulates a process that
    dies (or a machine that sleeps) partway through embedding a document."""

    name = "fake"
    dim = EMBEDDING_DIM

    def __init__(self) -> None:
        self.batches = 0

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.batches += 1
        if self.batches >= 2:
            raise RuntimeError("simulated interruption during embedding")
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dim


def test_ingestion_resumes_embedding_after_an_interrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.services.ingestion.EMBEDDING_BATCH_SIZE", 2)
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        database_url = pg.get_connection_url()
        command.upgrade(_alembic_config(database_url), "head")
        asyncio.run(_interrupt_then_resume(database_url, tmp_path))


async def _interrupt_then_resume(database_url: str, tmp_path: Path) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    content = _build_test_pdf(tmp_path)

    async with session_factory() as session:
        tenant = Tenant(name="Demo Tenant")
        session.add(tenant)
        await session.commit()
        tenant_id = tenant.id

    async with session_factory() as session:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            await ingest_document(
                session, _EmbedderFailingOnSecondBatch(), tenant_id, "test.pdf", "norm", content
            )

    async with session_factory() as session:
        document = await session.scalar(select(Document))
        assert document is not None and document.status == "failed"
        total_chunks = await session.scalar(select(func.count()).select_from(Chunk))
        embedded = await session.scalar(select(func.count()).select_from(Embedding))
        assert total_chunks is not None and total_chunks > 2
        assert embedded == 2  # only the first batch was committed

    async with session_factory() as session:
        document, already_ingested = await ingest_document(
            session, _FakeEmbedder(), tenant_id, "test.pdf", "norm", content
        )
        assert already_ingested is False
        assert document.status == "ready"

    async with session_factory() as session:
        total_chunks = await session.scalar(select(func.count()).select_from(Chunk))
        embedded = await session.scalar(select(func.count()).select_from(Embedding))
        assert embedded == total_chunks

    await engine.dispose()
