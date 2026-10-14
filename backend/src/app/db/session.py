from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, SessionTransaction

from app.core.config import settings

engine: AsyncEngine = create_async_engine(settings.database_url, pool_pre_ping=True)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

APP_ROLE = "normeon_app"  # created by migration 0004
_TENANT_KEY = "tenant_id"


@event.listens_for(Session, "after_begin")
def _enter_tenant_scope(session: Session, transaction: SessionTransaction, connection: Any) -> None:
    """Every transaction of a tenant-scoped session starts as `normeon_app`
    with `app.tenant_id` set, so row-level security applies to everything
    it runs (migration 0004). Both are `LOCAL`: they end with the
    transaction, so a pooled connection never carries a tenant over to the
    next request. Re-applied on each new transaction, which is why this is
    a listener rather than a one-off statement at session start: services
    commit mid-request (ingestion commits per status change)."""
    tenant_id = session.info.get(_TENANT_KEY)
    if tenant_id is None:
        return
    connection.exec_driver_sql(f"SET LOCAL ROLE {APP_ROLE}")
    connection.execute(
        text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
        {"tenant_id": str(tenant_id)},
    )


async def get_session() -> AsyncIterator[AsyncSession]:
    """An unscoped session: for login, health and maintenance scripts only.
    Request handlers that touch tenant data use `tenant_session`."""
    async with async_session_factory() as session:
        yield session


@asynccontextmanager
async def tenant_session(
    tenant_id: UUID, factory: async_sessionmaker[AsyncSession] = async_session_factory
) -> AsyncIterator[AsyncSession]:
    """A session whose every transaction is confined to `tenant_id` by
    row-level security, on top of the queries' own tenant filters."""
    async with factory(info={_TENANT_KEY: tenant_id}) as session:
        yield session
