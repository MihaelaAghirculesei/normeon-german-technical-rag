"""One real Postgres for the whole security suite: migrations applied
(including 0004's role and policies), three tenants, one user each, and a
document that only tenant B owns.

  A  anna  admin   no documents      -- asks about B's content
  B  ben   reader  one ingested PDF  -- control: the content is findable
  C  carla admin   no documents      -- uploads through the API

The app runs against it through dependency overrides, with a hashing
embedder, the no-op reranker and the fake LLM: deterministic and offline,
but every query and write goes through the real database, as the
`normeon_app` role, under row-level security.
"""

import asyncio
import functools
import hashlib
import math
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from testcontainers.community.postgres import PostgresContainer

from app.adapters.llm.fake import FakeLlmClient
from app.adapters.reranker.noop import NoopReranker
from app.api.deps import (
    AuthDep,
    get_embedder,
    get_llm_client,
    get_reranker,
    get_session,
    get_tenant_session,
)
from app.api.v1 import documents
from app.core.security import hash_password
from app.db.models import Tenant, User
from app.db.session import tenant_session
from app.main import app
from app.services.ingestion import ingest_document

MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"
PASSWORD = "sicheres-testpasswort"

TENANT_A = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000000")
TENANT_B = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000000")
TENANT_C = uuid.UUID("cccccccc-0000-0000-0000-000000000000")
USERS = {
    "anna@a.test": (TENANT_A, "admin"),
    "ben@b.test": (TENANT_B, "reader"),
    "carla@c.test": (TENANT_C, "admin"),
}

# Only tenant B's document talks about this.
B_SECTIONS = [
    (
        "4 Notbremsassistent",
        "Der Notbremsassistent muss bei einer Annaeherungsgeschwindigkeit von "
        "fuenfzig Kilometern pro Stunde eine Kollision mit einem stehenden "
        "Hindernis vollstaendig vermeiden. " * 8,
    ),
]
B_QUESTION = "Welche Annaeherungsgeschwindigkeit muss der Notbremsassistent beherrschen?"


class HashingEmbedder:
    """Deterministic bag-of-words embedder, as in the integration tests."""

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


def build_pdf(path: Path, sections: list[tuple[str, str]]) -> bytes:
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


@dataclass(frozen=True)
class SecurityDb:
    url: str
    b_document_id: uuid.UUID

    def factory(self) -> async_sessionmaker[AsyncSession]:
        # NullPool: no connection outlives the event loop that opened it
        # (TestClient and asyncio.run each bring their own).
        return async_sessionmaker(
            create_async_engine(self.url, poolclass=NullPool), expire_on_commit=False
        )


async def _seed(url: str, tmp: Path) -> uuid.UUID:
    engine = create_async_engine(url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        for tenant_id, name in [(TENANT_A, "A"), (TENANT_B, "B"), (TENANT_C, "C")]:
            session.add(Tenant(id=tenant_id, name=name))
        await session.flush()
        for email, (tenant_id, role) in USERS.items():
            session.add(
                User(
                    tenant_id=tenant_id,
                    email=email,
                    password_hash=hash_password(PASSWORD),
                    role=role,
                )
            )
        await session.commit()

        document, _ = await ingest_document(
            session,
            HashingEmbedder(),
            TENANT_B,
            "Lastenheft-B.pdf",
            "lastenheft",
            build_pdf(tmp / "b.pdf", B_SECTIONS),
        )
    await engine.dispose()
    return document.id


@pytest.fixture(scope="module")
def security_db(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SecurityDb]:
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        url = pg.get_connection_url()
        cfg = Config()
        cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
        cfg.set_main_option("sqlalchemy.url", url)
        command.upgrade(cfg, "head")
        b_document_id = asyncio.run(_seed(url, tmp_path_factory.mktemp("security")))
        yield SecurityDb(url=url, b_document_id=b_document_id)


@pytest.fixture
def client(security_db: SecurityDb, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    factory = security_db.factory()

    async def session_override() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    async def tenant_session_override(auth: AuthDep) -> AsyncIterator[AsyncSession]:
        async with tenant_session(auth.tenant_id, factory) as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_tenant_session] = tenant_session_override
    app.dependency_overrides[get_embedder] = HashingEmbedder
    app.dependency_overrides[get_reranker] = NoopReranker
    app.dependency_overrides[get_llm_client] = FakeLlmClient
    # Background ingestion opens its own tenant session; point it here too.
    monkeypatch.setattr(
        documents, "tenant_session", functools.partial(tenant_session, factory=factory)
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def login(client: TestClient, email: str, password: str = PASSWORD) -> dict[str, str]:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
