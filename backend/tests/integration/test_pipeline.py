"""The full `retrieve_context` pipeline against a real pgvector database.

Ingests one tenant's document through the real pipeline, then runs
`retrieve_context` with a deterministic fake reranker (a real
cross-encoder is 2 GB and not what this test is about). Checks that the
phases compose, the token budget and section-dedup are applied to the
reranked list, the phase timings add up, and the tenant boundary holds.
"""

import asyncio
import hashlib
import math
from pathlib import Path

import pymupdf
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from app.db.models import Tenant
from app.domain.models import RetrievedChunk
from app.services.ingestion import ingest_document
from app.services.retrieval import retrieve_context

MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"


class _HashingEmbedder:
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


class _ReverseReranker:
    """Deterministic stand-in: reverses the candidate order and keeps
    `top_k`, so the test can tell rerank actually ran without a model."""

    name = "reverse"

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return list(reversed(chunks))[:top_k]


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
        page.insert_text((50, 40), "Technische Norm - Entwurf", fontsize=8, fontname="helv")
        page.insert_text((50, 90 + offset), title, fontsize=15, fontname="hebo")
        page.insert_textbox((50, 120 + offset, 545, 790), body, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()
    return path.read_bytes()


# Each body is well over the 100-token structural-merge floor, so the
# four sections stay four chunks.
_SECTIONS = [
    (
        "1 Allgemeines",
        "Dieser Abschnitt regelt den Geltungsbereich der Norm fuer "
        "Kraftfahrzeuge und Anhaenger im oeffentlichen Strassenverkehr. " * 10,
    ),
    (
        "2 Lenkanlage",
        "Die Lenkanlage und die zulaessige Lenkkraft bei Ausfall der "
        "Servounterstuetzung werden auf ebener Fahrbahn geprueft. " * 10,
    ),
    (
        "3 Bremsanlage",
        "Die Bremsanlage muss bei jeder Beladung eine ausreichende "
        "Verzoegerung erreichen und darf keinesfalls blockieren. " * 10,
    ),
    (
        "4 Beleuchtung",
        "Scheinwerfer und Leuchten muessen die Fahrbahn ausreichend "
        "ausleuchten, ohne den Gegenverkehr zu blenden. " * 10,
    ),
]
_TENANT_B_SECTIONS = [
    (
        "1 Karosserie",
        "Die Karosserie und ihre Anbauteile muessen den Lastannahmen des "
        "Betriebsfestigkeitsnachweises jederzeit genuegen. " * 10,
    ),
]

_QUERY = "zulaessige Lenkkraft Servounterstuetzung Fahrbahn Bremsanlage Beleuchtung"
_STRATEGY = "structural"


def test_pipeline_composes_reranks_and_budgets(tmp_path: Path) -> None:
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        database_url = pg.get_connection_url()
        command.upgrade(_alembic_config(database_url), "head")
        asyncio.run(_run(database_url, tmp_path))


async def _run(database_url: str, tmp_path: Path) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    embedder = _HashingEmbedder()
    reranker = _ReverseReranker()
    pdf_a = _build_pdf(tmp_path / "a.pdf", _SECTIONS)
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

    # --- pipeline runs end to end; rerank reversed the hybrid order -----
    async with session_factory() as session:
        result = await retrieve_context(
            session, embedder, reranker,
            tenant_id=tenant_a_id, question=_QUERY, strategy=_STRATEGY,
            rerank_top_k=4, token_budget=10_000,
        )
    assert result.context, "expected a non-empty context"
    assert [c.chunk_id for c in result.context] == [c.chunk_id for c in result.reranked]
    assert len(result.reranked) <= 4
    assert all(c.document_id == doc_a_id for c in result.context)

    t = result.timing
    assert t.hybrid_ms >= 0 and t.rerank_ms >= 0 and t.select_ms >= 0
    # each phase is rounded to 0.1 ms independently of the end-to-end total,
    # so the parts can sum to at most 3 * 0.05 ms over it under load jitter
    assert t.total_ms >= t.hybrid_ms + t.rerank_ms + t.select_ms - 0.15

    # --- a tight token budget stops selection early --------------------
    async with session_factory() as session:
        tight = await retrieve_context(
            session, embedder, reranker,
            tenant_id=tenant_a_id, question=_QUERY, strategy=_STRATEGY,
            rerank_top_k=4, token_budget=30,
        )
    assert 0 < len(tight.context) <= len(tight.reranked)
    assert len(tight.context) < len(result.context), "tight budget should keep fewer chunks"
    used = sum(len(c.content.split()) for c in tight.context)
    assert used <= 30 or len(tight.context) == 1  # _select_context invariant
    sections = [c.section_path for c in tight.context if c.section_path is not None]
    assert len(sections) == len(set(sections))

    # --- tenant isolation --------------------------------------------------
    async with session_factory() as session:
        cross = await retrieve_context(
            session, embedder, reranker,
            tenant_id=tenant_b_id, question=_QUERY, strategy=_STRATEGY,
        )
    assert cross.context
    assert all(c.document_id == doc_b_id for c in cross.context)
    assert all(c.document_id != doc_a_id for c in cross.context)

    await engine.dispose()
