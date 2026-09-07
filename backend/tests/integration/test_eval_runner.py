"""Day 16 "Fatto quando": eval/runner.run_evaluation runs end to end on
three questions against a real pgvector database (real ingestion +
retrieval, deterministic hashing embedder + passthrough reranker +
FakeLlmClient) and produces a valid EvalReport.
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

from app.adapters.llm.fake import FakeLlmClient
from app.db.models import Tenant
from app.domain.models import RetrievedChunk
from app.eval.models import EvalConfig, EvalQuestion
from app.eval.runner import run_evaluation, write_report
from app.services.ingestion import ingest_document

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


def _build_pdf(path: Path) -> bytes:
    doc = pymupdf.open()
    sections = [
        ("1 Lenkanlage", "Die zulaessige Lenkkraft bei Ausfall der Servounterstuetzung "
                         "wird auf ebener Fahrbahn geprueft. " * 12),
        ("2 Bremsanlage", "Die Bremsanlage muss bei jeder Beladung eine ausreichende "
                          "Verzoegerung erreichen und darf nicht blockieren. " * 12),
    ]
    for page_no, (title, body) in enumerate(sections, start=1):
        page = doc.new_page()
        # A per-page vertical offset so the titles don't land at the same
        # bbox on every page -- the Day 3 parser filters repeated-bbox
        # blocks as headers/footers.
        offset = page_no * 20
        page.insert_text((50, 40), "Technische Norm - Entwurf", fontsize=8, fontname="helv")
        page.insert_text((50, 90 + offset), title, fontsize=15, fontname="hebo")
        page.insert_textbox((50, 120 + offset, 545, 790), body, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()
    return path.read_bytes()


_QUESTIONS = [
    EvalQuestion.model_validate(
        {"id": "Q001", "category": "requirement_lookup",
         "question": "Wie wird die zulaessige Lenkkraft geprueft?"}
    ),
    EvalQuestion.model_validate(
        {"id": "Q002", "category": "code_lookup",
         "question": "Was muss die Bremsanlage bei jeder Beladung erreichen?"}
    ),
    EvalQuestion.model_validate(
        {"id": "Q003", "category": "unanswerable", "should_abstain": True,
         "question": "Wie hoch ist der Oelpreis in Katar?"}
    ),
]

_CONFIG = EvalConfig(
    chunking_strategy="fixed_500", reranker="noop", top_k=5,
    embedding_provider="hash_test", llm_provider="fake", llm_model="",
)


def test_run_evaluation_end_to_end(tmp_path: Path) -> None:
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        database_url = pg.get_connection_url()
        command.upgrade(_alembic_config(database_url), "head")
        asyncio.run(_run(database_url, tmp_path))


async def _run(database_url: str, tmp_path: Path) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    embedder = _HashingEmbedder()
    pdf = _build_pdf(tmp_path / "norm.pdf")

    async with session_factory() as session:
        tenant = Tenant(name="Eval Tenant")
        session.add(tenant)
        await session.commit()
        tenant_id = tenant.id

    async with session_factory() as session:
        await ingest_document(session, embedder, tenant_id, "norm.pdf", "norm", pdf)

    report = await run_evaluation(
        _QUESTIONS, _CONFIG,
        session_factory=session_factory, embedder=embedder,
        reranker=_PassthroughReranker(), llm=FakeLlmClient(), tenant_id=tenant_id,
    )

    assert report.n_questions == 3
    assert [r.question_id for r in report.results] == ["Q001", "Q002", "Q003"]
    assert all(r.error is None for r in report.results)
    assert len(report.config_hash) == 64
    answered = [r for r in report.results if not r.abstained]
    assert answered, "at least one question should have produced a cited answer"
    assert all(r.retrieved for r in answered)

    path = write_report(report, tmp_path / "reports")
    assert path.name == f"{report.config_hash}.json"

    await engine.dispose()
