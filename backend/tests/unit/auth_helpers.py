"""Shared by the endpoint tests: stand in for authentication and the
tenant-scoped session, so each test exercises one endpoint's contract
without a token round trip or a database."""

import uuid
from typing import Any

from fastapi import FastAPI

from app.api.deps import get_auth, get_tenant_session
from app.core.security import AuthContext, Role

TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def auth_context(role: Role = "admin", tenant_id: uuid.UUID = TENANT_ID) -> AuthContext:
    return AuthContext(
        user_id=uuid.UUID(int=7), tenant_id=tenant_id, role=role, email="anna@demo.normeon.dev"
    )


def authenticate(app: FastAPI, session: Any, role: Role = "admin") -> None:
    """Every request is made by a user of TENANT_ID with `role`, and gets
    `session` as its tenant session."""
    app.dependency_overrides[get_auth] = lambda: auth_context(role)
    app.dependency_overrides[get_tenant_session] = lambda: session
