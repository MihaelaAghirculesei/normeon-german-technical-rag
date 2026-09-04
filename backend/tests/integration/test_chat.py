"""The Day 11 acceptance check against a real pgvector database:
a question produces a German answer carrying `[S1]` markers, and every
marker in that answer resolves to a real source with real page metadata.

Ingestion and retrieval are the real pipeline; the reranker is a
deterministic stand-in and the LLM is the FakeLlmClient (which cites
whatever markers the rendered prompt puts in front of it -- enough to
prove the wiring without a key).
"""

import asyncio
import hashlib
import math
import re
from pathlib import Path

import pymupdf
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from app.adapters.llm.fake import FakeLlmClient
from app.db.models import Tenant
from app.domain.models import RetrievedChunk
from app.services.generation import generate_answer
from app.services.ingestion import ingest_document

MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"
_MARKER = re.compile(r"\[S\d+\]")


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


class _PassthroughReranker:
    name = "passthrough"

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return chunks[:top_k]


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
]
_TENANT_B_SECTIONS = [
    (
        "1 Karosserie",
        "Die Karosserie und ihre Anbauteile muessen den Lastannahmen des "
        "Betriebsfestigkeitsnachweises jederzeit genuegen. " * 10,
    ),
]

_QUERY = "zulaessige Lenkkraft Servounterstuetzung Fahrbahn Bremsanlage"
_STRATEGY = "structural"


def test_chat_produces_a_german_answer_with_resolvable_markers(tmp_path: Path) -> None:
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        database_url = pg.get_connection_url()
        command.upgrade(_alembic_config(database_url), "head")
        asyncio.run(_run(database_url, tmp_path))


async def _run(database_url: str, tmp_path: Path) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    embedder = _HashingEmbedder()
    reranker = _PassthroughReranker()
    llm = FakeLlmClient()
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

    # --- the acceptance check --------------------------------------------
    async with session_factory() as session:
        result = await generate_answer(
            session, embedder, reranker, llm,
            tenant_id=tenant_a_id, question=_QUERY, strategy=_STRATEGY,
        )

    assert result.answer != "NICHT_GEFUNDEN"
    assert "[S1]" in result.answer
    assert result.prompt_name == "answer_de.v1"
    assert len(result.prompt_sha256) == 64
    assert result.model == "fake"

    # markers are contiguous S1..Sn and every one resolves to a real source
    markers_in_answer = {m.strip("[]") for m in _MARKER.findall(result.answer)}
    by_marker = {s.marker for s in result.sources}
    assert by_marker == {f"S{i}" for i in range(1, len(result.sources) + 1)}
    assert markers_in_answer <= by_marker
    assert result.sources, "expected at least one source"
    for src in result.sources:
        assert src.page_from >= 1 and src.page_to >= src.page_from
        assert src.filename == "a.pdf"
        assert src.document_id == doc_a_id

    # every marker the fake actually cited was validated: no invented
    # marker survives, and each citation resolves to the same real chunk
    assert result.citations, "expected at least one validated citation"
    assert {c.marker for c in result.citations} == markers_in_answer
    for citation in result.citations:
        assert citation.document_id == doc_a_id
        assert citation.snippet

    # --- tenant isolation: tenant B never sees tenant A's document -------
    async with session_factory() as session:
        cross = await generate_answer(
            session, embedder, reranker, llm,
            tenant_id=tenant_b_id, question=_QUERY, strategy=_STRATEGY,
        )
    assert all(s.document_id == doc_b_id for s in cross.sources)
    assert all(s.document_id != doc_a_id for s in cross.sources)

    await engine.dispose()
