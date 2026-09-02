"""Hybrid (RRF) retrieval against a real pgvector database.

Ingests one tenant's document through the real pipeline, then checks that
`hybrid_search`:
- fuses exactly the three DB branches (vector, full-text, trigram) with
  RRF and re-hydrates the winning ids -- verified by reproducing the
  fusion from the branch outputs and comparing;
- blends the branches rather than passing one of them through;
- keeps the trigram branch in play for a code query;
- never crosses the tenant boundary and returns fused scores in order.

The "hybrid surfaces a chunk neither branch ranked on its own" property
is shown on the real corpus by scripts/try_hybrid_search.py.
"""

import asyncio
import hashlib
import math
from pathlib import Path

import pymupdf
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from app.db.models import Tenant
from app.domain.fusion import rrf
from app.services.ingestion import ingest_document
from app.services.retrieval import (
    _fts_branch,
    _trgm_branch,
    hybrid_search,
    vector_search,
)

MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"


class _HashingEmbedder:
    """Deterministic bag-of-words embedder (same shape as the one in
    test_vector_search): tokens hash into `dim` buckets, vector is
    L2-normalised. Overlap in vocabulary -> high cosine similarity, so a
    section that shares more total words with the query ranks higher --
    which is exactly where it diverges from `ts_rank_cd`'s proximity
    weighting, giving the two branches different leaders to fuse.
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


# A spread of sections so each branch has a non-trivial candidate list to
# fuse: prose that overlaps the query to varying degrees, a code section,
# and unrelated filler.
_SECTIONS = [
    (
        "1 Allgemeines",
        "Dieser Abschnitt regelt den Geltungsbereich der Norm und nennt die "
        "allgemeinen Randbedingungen fuer Kraftfahrzeuge und Anhaenger. " * 5,
    ),
    (
        "2 Lenkanlage",
        "Die Lenkanlage steuert die Fahrtrichtung. Die zulaessige Lenkkraft "
        "haengt von der Servounterstuetzung ab. Faellt die Servounterstuetzung "
        "aus, steigt die noetige Lenkkraft am Lenkrad deutlich an. Die "
        "Lenkkraft, die Servounterstuetzung und das Verhalten auf nasser "
        "Fahrbahn werden bewertet. Weitere Hinweise zur Lenkkraft und zur "
        "Servounterstuetzung folgen im Abschnitt Pruefung. " * 4,
    ),
    (
        "3 Pruefverfahren",
        "Fuer den Nachweis gilt: die Lenkkraft bei ausgefallener "
        "Servounterstuetzung wird auf ebener Fahrbahn im Rahmen der Pruefung "
        "gemessen. "
        + "Der Pruefstand erfasst Temperatur, Luftdruck und Beladung des "
        "Fahrzeugs sowie den Zustand der Bereifung und der Radaufhaengung. " * 5,
    ),
    (
        "4 Anforderung LH-3.2.1",
        "Die Anforderung LH-3.2.1 legt fest, dass die Betaetigungskraft bei "
        "Ausfall der Servounterstuetzung 300 N nicht ueberschreiten darf. "
        "LH-3.2.1 verweist auf das Pruefverfahren in Abschnitt 3. " * 4,
    ),
    (
        "5 Bremsanlage",
        "Die Bremsanlage muss bei jeder Beladung eine ausreichende "
        "Verzoegerung erreichen und darf nicht blockieren. " * 5,
    ),
    (
        "6 Beleuchtung",
        "Scheinwerfer und Leuchten muessen die Fahrbahn ausreichend "
        "ausleuchten, ohne den Gegenverkehr zu blenden. " * 5,
    ),
]
_TENANT_B_SECTIONS = [
    (
        "1 Karosserie",
        "Die Karosserie und ihre Anbauteile muessen den Lastannahmen des "
        "Betriebsfestigkeitsnachweises genuegen. " * 6,
    ),
]

_PROSE_QUERY = "zulaessige Lenkkraft Servounterstuetzung Fahrbahn Pruefung"
_CODE_QUERY = "Welche Betaetigungskraft nennt LH-3.2.1?"
_STRATEGY = "structural"


def test_hybrid_search_fuses_branches_and_isolates_tenants(tmp_path: Path) -> None:
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        database_url = pg.get_connection_url()
        command.upgrade(_alembic_config(database_url), "head")
        asyncio.run(_run(database_url, tmp_path))


async def _run(database_url: str, tmp_path: Path) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    embedder = _HashingEmbedder()
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

    # --- hybrid_search == RRF over exactly the three DB branches --------
    #     Reproduce the fusion from the branch outputs and compare, so the
    #     candidate pull, the id-level fusion and the re-hydration are all
    #     pinned without depending on which chunk each branch ranks first.
    async with session_factory() as session:
        vec = await vector_search(
            session, embedder, tenant_id=tenant_a_id, question=_PROSE_QUERY,
            strategy=_STRATEGY, k=10,
        )
    async with session_factory() as session:
        fts_branch = await _fts_branch(
            session, tenant_id=tenant_a_id, question=_PROSE_QUERY, strategy=_STRATEGY, k=10
        )
        trgm_branch = await _trgm_branch(
            session, tenant_id=tenant_a_id, question=_PROSE_QUERY, strategy=_STRATEGY, k=10
        )
    async with session_factory() as session:
        hybrid = await hybrid_search(
            session, embedder, tenant_id=tenant_a_id, question=_PROSE_QUERY,
            strategy=_STRATEGY, candidate_k=10, top_k=5,
        )

    expected = rrf(
        [
            [str(h.chunk_id) for h in vec],
            [str(h.chunk_id) for h in fts_branch],
            [str(h.chunk_id) for h in trgm_branch],
        ],
        k=60,
        weights=[1.0, 1.0, 1.0],
    )[:5]
    assert [str(h.chunk_id) for h in hybrid] == [chunk_id for chunk_id, _ in expected]
    assert [h.score for h in hybrid] == pytest.approx([score for _, score in expected])
    assert all(h.document_id == doc_a_id for h in hybrid)

    # fused scores come back ordered and in the RRF range
    fused_scores = [h.score for h in hybrid]
    assert fused_scores == sorted(fused_scores, reverse=True)
    assert all(0.0 < s < 0.2 for s in fused_scores)

    # --- code query: the trigram branch keeps the LH-3.2.1 chunk in the
    #     fused result ------------------------------------------------------
    async with session_factory() as session:
        code_hybrid = await hybrid_search(
            session, embedder, tenant_id=tenant_a_id, question=_CODE_QUERY,
            strategy=_STRATEGY, candidate_k=10, top_k=5,
        )
    assert any("LH-3.2.1" in h.content for h in code_hybrid)

    # --- tenant isolation ---------------------------------------------------
    async with session_factory() as session:
        cross = await hybrid_search(
            session, embedder, tenant_id=tenant_b_id, question=_CODE_QUERY,
            strategy=_STRATEGY, candidate_k=10, top_k=10,
        )
    assert cross, "tenant B has its own document and should still match something"
    assert all(h.document_id == doc_b_id for h in cross)
    assert all(h.document_id != doc_a_id for h in cross)

    await engine.dispose()
