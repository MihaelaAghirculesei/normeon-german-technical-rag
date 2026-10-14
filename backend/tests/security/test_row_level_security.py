"""Row-level security on its own, below the application (migration 0004).

These run SQL with no tenant filter at all -- the mistake RLS exists to
catch -- through `tenant_session`, and check the database itself confines
reads and writes to the session's tenant, and fails closed without one.
"""

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.db.session import tenant_session
from tests.security.conftest import TENANT_A, TENANT_B, SecurityDb

_COUNT = "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"


def _counts(db: SecurityDb, tenant: uuid.UUID | None) -> tuple[int, int]:
    async def run() -> tuple[int, int]:
        factory = db.factory()
        if tenant is None:
            async with factory() as session:  # unscoped: the connecting superuser
                row = (await session.execute(text(_COUNT))).one()
        else:
            async with tenant_session(tenant, factory) as session:
                row = (await session.execute(text(_COUNT))).one()
        return int(row[0]), int(row[1])

    return asyncio.run(run())


def test_an_unfiltered_query_only_sees_the_sessions_tenant(security_db: SecurityDb) -> None:
    all_docs, all_chunks = _counts(security_db, None)
    b_docs, b_chunks = _counts(security_db, TENANT_B)

    assert b_docs >= 1 and b_chunks >= 1
    assert (b_docs, b_chunks) <= (all_docs, all_chunks)
    assert _counts(security_db, TENANT_A) == (0, 0)


def test_the_policy_holds_on_every_new_transaction(security_db: SecurityDb) -> None:
    """Services commit mid-request; the scope must survive a commit."""

    async def run() -> list[int]:
        seen = []
        async with tenant_session(TENANT_A, security_db.factory()) as session:
            for _ in range(2):
                seen.append(int(await session.scalar(text("SELECT count(*) FROM chunks")) or 0))
                await session.commit()
        return seen

    assert asyncio.run(run()) == [0, 0]


def test_writing_a_row_for_another_tenant_is_rejected(security_db: SecurityDb) -> None:
    async def run() -> None:
        async with tenant_session(TENANT_A, security_db.factory()) as session:
            await session.execute(
                text(
                    "INSERT INTO documents (tenant_id, content_hash, filename, doc_type) "
                    "VALUES (:tenant, :hash, 'eingeschleust.pdf', 'norm')"
                ),
                {"tenant": TENANT_B, "hash": "f" * 64},
            )

    with pytest.raises(DBAPIError, match="row-level security"):
        asyncio.run(run())


def test_moving_a_row_to_another_tenant_is_rejected(security_db: SecurityDb) -> None:
    async def run() -> None:
        async with tenant_session(TENANT_B, security_db.factory()) as session:
            await session.execute(
                text("UPDATE documents SET tenant_id = :other"), {"other": TENANT_A}
            )

    with pytest.raises(DBAPIError, match="row-level security"):
        asyncio.run(run())


def test_without_a_tenant_the_app_role_sees_nothing(security_db: SecurityDb) -> None:
    async def run() -> tuple[int, int]:
        async with security_db.factory()() as session:
            await session.execute(text("SET LOCAL ROLE normeon_app"))
            row = (await session.execute(text(_COUNT))).one()
            return int(row[0]), int(row[1])

    assert asyncio.run(run()) == (0, 0)


def test_the_tenant_scope_does_not_leak_into_the_next_transaction(
    security_db: SecurityDb,
) -> None:
    async def run() -> tuple[str | None, str]:
        async with tenant_session(TENANT_B, security_db.factory()) as session:
            await session.execute(text("SELECT 1"))
            await session.commit()
            # A raw connection-level look after the scoped transaction ended.
            conn = await session.connection()
            await conn.exec_driver_sql("COMMIT")
            await conn.exec_driver_sql("BEGIN")
            tenant = await conn.scalar(text("SELECT current_setting('app.tenant_id', true)"))
            role = await conn.scalar(text("SELECT current_user"))
            return tenant, str(role)

    tenant, role = asyncio.run(run())
    assert role != "normeon_app"
    assert tenant in (None, "")
