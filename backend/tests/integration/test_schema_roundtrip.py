"""Applies the real Alembic migrations to a throwaway Postgres container,
then round-trips a chunk with an embedding through the ORM. If this
passes, the schema in migrations/versions/0001_initial_schema.py actually
works against a real pgvector database — not just against SQLAlchemy's
model definitions.
"""

import asyncio
import uuid
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from app.db.models import Chunk, Document, Embedding, Tenant

MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"


def _alembic_config(database_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


def test_migrations_apply_and_chunk_roundtrips() -> None:
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        database_url = pg.get_connection_url()
        command.upgrade(_alembic_config(database_url), "head")
        asyncio.run(_insert_and_read_back(database_url))


async def _insert_and_read_back(database_url: str) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    fake_vector = [0.001 * i for i in range(1024)]

    async with session_factory() as session:
        tenant = Tenant(name="Demo Tenant")
        session.add(tenant)
        await session.flush()

        document = Document(
            tenant_id=tenant.id,
            content_hash="a" * 64,
            filename="test.pdf",
            doc_type="norm",
        )
        session.add(document)
        await session.flush()

        chunk = Chunk(
            document_id=document.id,
            tenant_id=tenant.id,
            strategy="fixed_500",
            chunk_hash="b" * 64,
            ordinal=0,
            page_from=1,
            page_to=1,
            section_path="3.2.1",
            heading="Lenkkraft",
            char_start=0,
            char_end=12,
            content="Testinhalt",
            content_norm="testinhalt",
            token_count=2,
        )
        session.add(chunk)
        await session.flush()

        session.add(Embedding(chunk_id=chunk.id, model="e5-large", dim=1024, vec=fake_vector))
        await session.commit()
        chunk_id: uuid.UUID = chunk.id

    async with session_factory() as session:
        reread = await session.get(Chunk, chunk_id)
        assert reread is not None
        assert reread.content == "Testinhalt"
        assert reread.section_path == "3.2.1"

        embedding = await session.get(Embedding, (chunk_id, "e5-large"))
        assert embedding is not None
        assert len(embedding.vec) == 1024
        assert embedding.vec[1] == fake_vector[1]

    await engine.dispose()
