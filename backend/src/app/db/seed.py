"""Idempotent demo-data seed. Run with: python -m app.db.seed

Two tenants, so tenant isolation can be shown end to end: the demo tenant
owns the ingested corpus; the second tenant starts empty.

The demo tenant has an admin (`anna`) and a read-only guest. The guest and
the second tenant's reader (`ben`) are meant to be public: they share
`settings.demo_reader_password`. The admin can upload documents, so her
password comes from `settings.demo_admin_password`, which a public
deployment sets to a secret value.

Every run also brings the stored passwords in line with the settings, so
changing a password is a matter of changing the setting and restarting.
"""

import asyncio
import uuid
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.security import hash_password, verify_password
from app.db.models import Tenant, User
from app.db.session import async_session_factory

DEMO_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
SECOND_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")

TENANTS = {
    DEMO_TENANT_ID: "Demo Tenant",
    SECOND_TENANT_ID: "Zulieferer GmbH",
}

Role = Literal["admin", "reader"]
# (email, tenant, role)
USERS: list[tuple[str, uuid.UUID, Role]] = [
    ("anna@demo.normeon.dev", DEMO_TENANT_ID, "admin"),
    ("gast@demo.normeon.dev", DEMO_TENANT_ID, "reader"),
    ("ben@zulieferer.normeon.dev", SECOND_TENANT_ID, "reader"),
]


async def seed(
    session_factory: async_sessionmaker[AsyncSession] = async_session_factory,
    *,
    admin_password: str | None = None,
    reader_password: str | None = None,
) -> None:
    passwords: dict[Role, str] = {
        "admin": admin_password or settings.demo_admin_password,
        "reader": reader_password or settings.demo_reader_password,
    }
    async with session_factory() as session:
        for tenant_id, name in TENANTS.items():
            if await session.get(Tenant, tenant_id) is None:
                session.add(Tenant(id=tenant_id, name=name))
                print(f"seeded tenant {name}: {tenant_id}")
        await session.flush()

        for email, tenant_id, role in USERS:
            password = passwords[role]
            existing = await session.scalar(select(User).where(User.email == email))
            if existing is None:
                session.add(
                    User(
                        tenant_id=tenant_id,
                        email=email,
                        password_hash=hash_password(password),
                        role=role,
                    )
                )
                print(f"seeded user {email} ({role})")
            elif not verify_password(password, existing.password_hash):
                existing.password_hash = hash_password(password)
                print(f"updated password of {email}")
            else:
                print(f"user already present: {email}")
        await session.commit()


if __name__ == "__main__":
    asyncio.run(seed())
