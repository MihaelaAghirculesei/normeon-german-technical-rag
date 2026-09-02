"""German full-text search against a real Postgres.

Covers the Day 7 acceptance criterion (a requirement-code query finds the
right chunk) plus the two behaviours normalize_de leans on / documents:
the `german` config folds umlauts itself, and it does *not* split
compounds (docs/FAILURE-MODES.md).
"""

import asyncio
from pathlib import Path

import pymupdf
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from app.db.models import Tenant
from app.services.ingestion import ingest_document
from app.services.retrieval import fts_search

MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"
EMBEDDING_DIM = 1024


class _FakeEmbedder:
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


def _build_pdf(path: Path, sections: list[tuple[str, str]]) -> bytes:
    doc = pymupdf.open()
    for page_no, (title, body) in enumerate(sections, start=1):
        page = doc.new_page()
        offset = page_no * 18
        page.insert_text((50, 40), "Lastenheft - Entwurf", fontsize=8, fontname="helv")
        page.insert_text((50, 90 + offset), title, fontsize=15, fontname="hebo")
        page.insert_textbox((50, 120 + offset, 545, 790), body, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()
    return path.read_bytes()


_SECTIONS_A = [
    (
        "1 Allgemeines",
        "Dieser Abschnitt beschreibt den Geltungsbereich und die "
        "allgemeinen Randbedingungen des Lastenhefts. " * 6,
    ),
    (
        "2 Lenkanlage",
        "Die Anforderung LH-3.2.1 legt fest, dass die Lenkkraft bei Ausfall "
        "der Servounterstuetzung 300 N nicht ueberschreiten darf. Die "
        "Pruefung erfolgt auf ebener Fahrbahn. " * 5,
    ),
    (
        "3 Bremsanlage",
        "Die Bremsanlage muss bei jeder Beladung eine ausreichende "
        "Verzoegerung sicherstellen und darf nicht blockieren. " * 6,
    ),
]
_SECTIONS_B = [
    (
        "1 Karosserie",
        "Die Karosserie und ihre Anbauteile muessen den Lastannahmen des "
        "Betriebsfestigkeitsnachweises genuegen. " * 6,
    ),
]


def test_fts_search_finds_a_requirement_code_and_isolates_tenants(tmp_path: Path) -> None:
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        database_url = pg.get_connection_url()
        command.upgrade(_alembic_config(database_url), "head")
        asyncio.run(_run(database_url, tmp_path))


async def _run(database_url: str, tmp_path: Path) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    embedder = _FakeEmbedder()
    pdf_a = _build_pdf(tmp_path / "a.pdf", _SECTIONS_A)
    pdf_b = _build_pdf(tmp_path / "b.pdf", _SECTIONS_B)

    async with session_factory() as session:
        tenant_a = Tenant(name="Tenant A")
        tenant_b = Tenant(name="Tenant B")
        session.add_all([tenant_a, tenant_b])
        await session.commit()
        tenant_a_id, tenant_b_id = tenant_a.id, tenant_b.id

    async with session_factory() as session:
        doc_a, _ = await ingest_document(
            session, embedder, tenant_a_id, "a.pdf", "lastenheft", pdf_a
        )
        doc_b, _ = await ingest_document(
            session, embedder, tenant_b_id, "b.pdf", "lastenheft", pdf_b
        )
        doc_a_id, doc_b_id = doc_a.id, doc_b.id

    # --- Day 7 acceptance: the code query lands on the right chunk, even
    #     though it's written "LH-3.2.1" in the doc and asked three ways ---
    for asked in ("Welche Anforderungen stellt LH-3.2.1?", "LH 3.2.1", "lh3.2.1"):
        async with session_factory() as session:
            hits = await fts_search(
                session, tenant_id=tenant_a_id, question=asked, strategy="structural", k=5
            )
        assert hits, f"no FTS hit for {asked!r}"
        assert "LH-3.2.1" in hits[0].content, f"wrong top chunk for {asked!r}"
        assert hits[0].document_id == doc_a_id

    # --- a plain prose query still works ---
    async with session_factory() as session:
        prose = await fts_search(
            session,
            tenant_id=tenant_a_id,
            question="Welche Verzoegerung muss die Bremsanlage sicherstellen?",
            strategy="structural",
            k=5,
        )
    assert prose
    assert "Bremsanlage" in prose[0].content

    # --- tenant isolation ---
    async with session_factory() as session:
        cross = await fts_search(
            session,
            tenant_id=tenant_b_id,
            question="Welche Anforderungen stellt LH-3.2.1?",
            strategy="structural",
            k=10,
        )
    assert all(h.document_id == doc_b_id for h in cross)
    assert all(h.document_id != doc_a_id for h in cross)

    # --- evidence for normalize_de's design choices ---
    async with session_factory() as session:
        umlaut_folds = await session.scalar(
            text("SELECT to_tsvector('german', 'Prüfung') = to_tsvector('german', 'Pruefung')")
        )
        compound_splits = await session.scalar(
            text(
                "SELECT to_tsvector('german', 'Fahrzeugzulassung') "
                "@@ websearch_to_tsquery('german', 'Zulassung')"
            )
        )
    assert umlaut_folds is True  # so normalize_de doesn't transliterate
    assert compound_splits is False  # documented in docs/FAILURE-MODES.md

    await engine.dispose()
