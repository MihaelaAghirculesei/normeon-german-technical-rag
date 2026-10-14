"""`get_chunk_context` against a real Postgres: neighbours come from the
same document and chunking strategy in `ordinal` order, the window is
clipped at the document's edges, and another tenant's chunk is
indistinguishable from a missing one.
"""

import asyncio
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from app.core.errors import ChunkNotFoundError
from app.db.models import Chunk, Document, Tenant
from app.services.chunks import get_chunk_context

MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"


def _alembic_config(database_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


def _chunk(document: Document, strategy: str, ordinal: int) -> Chunk:
    content = f"{strategy} {ordinal}"
    return Chunk(
        document_id=document.id,
        tenant_id=document.tenant_id,
        strategy=strategy,
        chunk_hash=f"{strategy}-{ordinal}".ljust(64, "0"),
        ordinal=ordinal,
        page_from=ordinal + 1,
        page_to=ordinal + 1,
        section_path=f"§ {ordinal}",
        heading=None,
        char_start=0,
        char_end=len(content),
        content=content,
        content_norm=content,
        token_count=2,
    )


def test_chunk_context_against_a_real_database() -> None:
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        database_url = pg.get_connection_url()
        command.upgrade(_alembic_config(database_url), "head")
        asyncio.run(_run(database_url))


async def _seed(
    session: AsyncSession,
) -> tuple[uuid.UUID, uuid.UUID, dict[tuple[str, int], uuid.UUID]]:
    tenant, other = Tenant(name="A"), Tenant(name="B")
    session.add_all([tenant, other])
    await session.flush()

    document = Document(
        tenant_id=tenant.id, content_hash="a" * 64, filename="StVZO.pdf", doc_type="norm"
    )
    elsewhere = Document(
        tenant_id=tenant.id, content_hash="b" * 64, filename="FZV.pdf", doc_type="norm"
    )
    session.add_all([document, elsewhere])
    await session.flush()

    chunks = [_chunk(document, "structural", i) for i in range(5)]
    chunks += [_chunk(document, "fixed_500", i) for i in range(3)]
    chunks += [_chunk(elsewhere, "structural", i) for i in range(3)]
    session.add_all(chunks)
    await session.commit()
    ids = {
        (c.strategy if c.document_id == document.id else "other", c.ordinal): c.id for c in chunks
    }
    return tenant.id, other.id, ids


async def _run(database_url: str) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        tenant_id, other_tenant_id, ids = await _seed(session)

    async with session_factory() as session:
        middle = await get_chunk_context(
            session, tenant_id=tenant_id, chunk_id=ids[("structural", 2)], window=1
        )
        assert middle.document.filename == "StVZO.pdf"
        assert middle.chunk.content == "structural 2"
        # Same document and strategy only: no fixed_500 window, no FZV chunk.
        assert [c.content for c in middle.before] == ["structural 1"]
        assert [c.content for c in middle.after] == ["structural 3"]

        wide = await get_chunk_context(
            session, tenant_id=tenant_id, chunk_id=ids[("structural", 2)], window=3
        )
        assert [c.ordinal for c in wide.before] == [0, 1]
        assert [c.ordinal for c in wide.after] == [3, 4]

        first = await get_chunk_context(
            session, tenant_id=tenant_id, chunk_id=ids[("structural", 0)], window=1
        )
        assert first.before == []
        assert [c.ordinal for c in first.after] == [1]

        alone = await get_chunk_context(
            session, tenant_id=tenant_id, chunk_id=ids[("fixed_500", 1)], window=0
        )
        assert (alone.before, alone.after) == ([], [])

        with pytest.raises(ChunkNotFoundError):
            await get_chunk_context(
                session, tenant_id=other_tenant_id, chunk_id=ids[("structural", 2)], window=1
            )
        with pytest.raises(ChunkNotFoundError):
            await get_chunk_context(session, tenant_id=tenant_id, chunk_id=uuid.uuid4(), window=1)

    await engine.dispose()
