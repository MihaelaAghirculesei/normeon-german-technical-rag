"""Vector search against a real pgvector database: ingest two tenants'
documents through the real pipeline, then assert that `vector_search`
ranks the relevant chunk first, never crosses the tenant boundary, honours
`k`, and copes when a selective filter leaves fewer than `k` matches (the
`hnsw.iterative_scan` path).
"""

import asyncio
import hashlib
import math
from pathlib import Path

import pymupdf
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from app.db.models import Chunk, Tenant
from app.services.ingestion import ingest_document
from app.services.retrieval import vector_search

MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"


class _HashingEmbedder:
    """Deterministic bag-of-words embedder: each token falls into one of
    `dim` buckets by hash, the vector is L2-normalised. Texts that share
    vocabulary get a high cosine similarity, so a query built from a
    section's words ranks that section's chunk first -- enough to test
    ranking, the tenant filter and `k` without loading a real model.
    """

    name = "hash_test"
    dim = 1024

    def _vec(self, text: str) -> list[float]:
        buckets = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            buckets[int.from_bytes(digest, "big") % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in buckets)) or 1.0
        return [x / norm for x in buckets]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _alembic_config(database_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


def _build_pdf(path: Path, sections: list[tuple[str, str]]) -> bytes:
    doc = pymupdf.open()
    for page_no, (title, body) in enumerate(sections, start=1):
        page = doc.new_page()
        offset = page_no * 18  # shift the body so the repeated-header filter spares it
        page.insert_text((50, 40), "Technische Norm - Entwurf", fontsize=8, fontname="helv")
        page.insert_text((50, 90 + offset), title, fontsize=15, fontname="hebo")
        page.insert_textbox((50, 120 + offset, 545, 790), body, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()
    return path.read_bytes()


_TENANT_A_SECTIONS = [
    (
        "1 Geltungsbereich",
        "Der Geltungsbereich dieser Verordnung umfasst alle Kraftfahrzeuge und "
        "Anhaenger im oeffentlichen Strassenverkehr. " * 8,
    ),
    (
        "2 Begriffsbestimmungen",
        "Begriffsbestimmungen und Definitionen der in dieser Verordnung "
        "verwendeten Fachbegriffe und Abkuerzungen folgen. " * 8,
    ),
    (
        "3 Lenkanlage",
        "Die Lenkanlage und die zulaessige Lenkkraft bei Ausfall der "
        "Servounterstuetzung unterliegen besonderen Anforderungen an die "
        "Betaetigungskraft. " * 8,
    ),
]
_TENANT_B_SECTIONS = [
    (
        "1 Bremsanlage",
        "Die Bremsanlage und die Bremswirkung jedes Fahrzeugs muessen den "
        "Vorschriften ueber die Verzoegerung entsprechen. " * 8,
    ),
]
_LENKANLAGE_QUERY = "Lenkanlage Lenkkraft Servounterstuetzung Betaetigungskraft"


def test_vector_search_ranks_isolates_and_limits(tmp_path: Path) -> None:
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        database_url = pg.get_connection_url()
        command.upgrade(_alembic_config(database_url), "head")
        asyncio.run(_run(database_url, tmp_path))


async def _run(database_url: str, tmp_path: Path) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    embedder = _HashingEmbedder()

    pdf_a = _build_pdf(tmp_path / "a.pdf", _TENANT_A_SECTIONS)
    pdf_b = _build_pdf(tmp_path / "b.pdf", _TENANT_B_SECTIONS)

    async with session_factory() as session:
        tenant_a = Tenant(name="Tenant A")
        tenant_b = Tenant(name="Tenant B")
        session.add_all([tenant_a, tenant_b])
        await session.commit()
        tenant_a_id, tenant_b_id = tenant_a.id, tenant_b.id

    async with session_factory() as session:
        doc_a, _ = await ingest_document(
            session, embedder, tenant_a_id, "a.pdf", "norm", pdf_a
        )
        doc_b, _ = await ingest_document(
            session, embedder, tenant_b_id, "b.pdf", "norm", pdf_b
        )
        doc_a_id, doc_b_id = doc_a.id, doc_b.id

    # --- ranking: the Lenkanlage section comes back first ---------------
    async with session_factory() as session:
        hits = await vector_search(
            session,
            embedder,
            tenant_id=tenant_a_id,
            question=_LENKANLAGE_QUERY,
            strategy="structural",
            k=3,
        )
    assert hits, "expected at least one hit"
    assert len(hits) <= 3
    assert "Lenk" in hits[0].content
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(-1e-6 <= h.score <= 1.0 + 1e-6 for h in hits)
    assert all(h.document_id == doc_a_id for h in hits)

    # --- k is honoured -------------------------------------------------
    async with session_factory() as session:
        two = await vector_search(
            session,
            embedder,
            tenant_id=tenant_a_id,
            question=_LENKANLAGE_QUERY,
            strategy="structural",
            k=2,
        )
    assert len(two) == 2

    # --- tenant isolation: B's search never sees A's chunks -----------
    async with session_factory() as session:
        b_hits = await vector_search(
            session,
            embedder,
            tenant_id=tenant_b_id,
            question=_LENKANLAGE_QUERY,
            strategy="structural",
            k=10,
        )
    assert b_hits, "tenant B has its own document, should still match something"
    assert all(h.document_id == doc_b_id for h in b_hits)
    assert all(h.document_id != doc_a_id for h in b_hits)

    # --- selective filter with fewer than k matches doesn't error and
    #     returns every structural chunk of the document ---------------
    async with session_factory() as session:
        structural_total = await session.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id == doc_a_id, Chunk.strategy == "structural")
        )
        many = await vector_search(
            session,
            embedder,
            tenant_id=tenant_a_id,
            question="Lenkkraft",
            strategy="structural",
            k=50,
        )
    assert structural_total is not None and structural_total < 50
    assert len(many) == structural_total

    await engine.dispose()
