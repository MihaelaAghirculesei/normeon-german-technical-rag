"""Idempotent demo-data seed. Run with: python -m app.db.seed

Only seeds a demo tenant. There is no `users` table yet — the schema (see
plan §5) doesn't define one, and Day 25 (auth) may turn out to need only
config-defined demo credentials tied to tenant_id rather than a full user
table. Revisit this file then.
"""

import asyncio
import uuid

from sqlalchemy import select

from app.db.models import Tenant
from app.db.session import async_session_factory

DEMO_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


async def seed() -> None:
    async with async_session_factory() as session:
        existing = await session.scalar(select(Tenant).where(Tenant.id == DEMO_TENANT_ID))
        if existing is not None:
            print(f"demo tenant already present: {existing.id}")
            return

        tenant = Tenant(id=DEMO_TENANT_ID, name="Demo Tenant")
        session.add(tenant)
        await session.commit()
        print(f"seeded demo tenant: {tenant.id}")


if __name__ == "__main__":
    asyncio.run(seed())
