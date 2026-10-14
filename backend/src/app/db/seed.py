"""Idempotent demo-data seed. Run with: python -m app.db.seed

Two tenants with one user each (Giorno 25), so tenant isolation can be
shown end to end: the demo tenant owns the ingested corpus; the second
tenant starts empty. Both users share `settings.demo_user_password`.
"""

import asyncio
import uuid

from sqlalchemy import select

from app.core.config import settings
from app.core.security import hash_password
from app.db.models import Tenant, User
from app.db.session import async_session_factory

DEMO_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
SECOND_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")

TENANTS = {
    DEMO_TENANT_ID: "Demo Tenant",
    SECOND_TENANT_ID: "Zulieferer GmbH",
}
# (email, tenant, role)
USERS = [
    ("anna@demo.normeon.dev", DEMO_TENANT_ID, "admin"),
    ("ben@zulieferer.normeon.dev", SECOND_TENANT_ID, "reader"),
]


async def seed() -> None:
    async with async_session_factory() as session:
        for tenant_id, name in TENANTS.items():
            if await session.get(Tenant, tenant_id) is None:
                session.add(Tenant(id=tenant_id, name=name))
                print(f"seeded tenant {name}: {tenant_id}")
        await session.flush()

        for email, tenant_id, role in USERS:
            existing = await session.scalar(select(User).where(User.email == email))
            if existing is not None:
                print(f"user already present: {email}")
                continue
            session.add(
                User(
                    tenant_id=tenant_id,
                    email=email,
                    password_hash=hash_password(settings.demo_user_password),
                    role=role,
                )
            )
            print(f"seeded user {email} ({role})")
        await session.commit()


if __name__ == "__main__":
    asyncio.run(seed())
