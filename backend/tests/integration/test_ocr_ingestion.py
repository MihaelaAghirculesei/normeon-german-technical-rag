"""A scanned page through the whole pipeline (Giorno 26): ingestion reads
it by OCR, the chunks built from it are flagged `ocr` in the database, and
retrieval hands the flag on -- which is what lets the UI say so."""

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from app.db.models import Chunk, Tenant
from app.services import ingestion
from app.services.ingestion import ingest_document
from app.services.retrieval import fts_search
from tests.integration.test_ingestion import _FakeEmbedder
from tests.unit.test_ocr import SCANNED_TEXT, FakeOcr, _pdf_with_a_scanned_page

MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"


def test_chunks_from_a_scanned_page_are_flagged_and_retrieved_as_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingestion, "default_ocr", FakeOcr)
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        url = pg.get_connection_url()
        cfg = Config()
        cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
        cfg.set_main_option("sqlalchemy.url", url)
        command.upgrade(cfg, "head")
        asyncio.run(_ingest_and_search(url, _pdf_with_a_scanned_page(tmp_path).read_bytes()))


async def _ingest_and_search(url: str, pdf: bytes) -> None:
    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        tenant = Tenant(name="Scan Tenant")
        session.add(tenant)
        await session.commit()
        await ingest_document(session, _FakeEmbedder(), tenant.id, "scan.pdf", "norm", pdf)

        fixed = (await session.scalars(select(Chunk).where(Chunk.strategy == "fixed_500"))).all()
        # The fixed window spans both pages: partly OCR text, so flagged.
        assert fixed and all(c.ocr for c in fixed)

        hits = await fts_search(
            session, tenant_id=tenant.id, question="Lenkkraft", strategy="structural", k=5
        )
        scanned = [h for h in hits if SCANNED_TEXT in h.content]
        assert scanned and all(h.ocr for h in scanned)
    await engine.dispose()
