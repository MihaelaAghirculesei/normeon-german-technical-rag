"""The demo seed against a real Postgres: who gets which role and password,
and that a re-run brings changed passwords in line without duplicating
anyone."""

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from app.core.security import verify_password
from app.db.models import User
from app.db.seed import DEMO_TENANT_ID, SECOND_TENANT_ID, seed

MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"


def test_seed_creates_the_guest_and_keeps_passwords_in_line() -> None:
    with PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg") as pg:
        url = pg.get_connection_url()
        cfg = Config()
        cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
        cfg.set_main_option("sqlalchemy.url", url)
        command.upgrade(cfg, "head")
        asyncio.run(_check(url))


async def _check(url: str) -> None:
    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def users() -> dict[str, User]:
        async with factory() as session:
            return {u.email: u for u in await session.scalars(select(User))}

    await seed(factory, admin_password="admin-secret", reader_password="public")
    first = await users()
    assert {email: (u.tenant_id, u.role) for email, u in first.items()} == {
        "anna@demo.normeon.dev": (DEMO_TENANT_ID, "admin"),
        "gast@demo.normeon.dev": (DEMO_TENANT_ID, "reader"),
        "ben@zulieferer.normeon.dev": (SECOND_TENANT_ID, "reader"),
    }
    assert verify_password("admin-secret", first["anna@demo.normeon.dev"].password_hash)
    assert not verify_password("public", first["anna@demo.normeon.dev"].password_hash)
    assert verify_password("public", first["gast@demo.normeon.dev"].password_hash)

    # Same settings: nothing is rewritten.
    await seed(factory, admin_password="admin-secret", reader_password="public")
    second = await users()
    assert {e: u.password_hash for e, u in second.items()} == {
        e: u.password_hash for e, u in first.items()
    }

    # A new admin password replaces the old one; the readers stay as they were.
    await seed(factory, admin_password="rotated", reader_password="public")
    third = await users()
    assert len(third) == 3
    assert verify_password("rotated", third["anna@demo.normeon.dev"].password_hash)
    assert not verify_password("admin-secret", third["anna@demo.normeon.dev"].password_hash)
    guest = "gast@demo.normeon.dev"
    assert third[guest].password_hash == first[guest].password_hash

    await engine.dispose()
