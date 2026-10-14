"""POST /api/v1/auth/login and GET /api/v1/auth/me (Giorno 25)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import AuthDep, DbSession
from app.core.errors import InvalidCredentialsError
from app.core.security import (
    DUMMY_PASSWORD_HASH,
    AuthContext,
    create_access_token,
    verify_password,
)
from app.db.models import Tenant, User

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)


class UserOut(BaseModel):
    user_id: UUID
    email: str
    role: Literal["admin", "reader"]
    tenant_id: UUID
    tenant_name: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: datetime
    user: UserOut


async def _user_out(session: DbSession, auth: AuthContext) -> UserOut:
    tenant = await session.get(Tenant, auth.tenant_id)
    return UserOut(
        user_id=auth.user_id,
        email=auth.email,
        role=auth.role,
        tenant_id=auth.tenant_id,
        tenant_name=tenant.name if tenant else "",
    )


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest, session: DbSession) -> LoginResponse:
    user = await session.scalar(select(User).where(User.email == payload.email.strip().lower()))
    # Hash even for an unknown email, so timing doesn't reveal which
    # accounts exist; the error is the same either way.
    valid = verify_password(payload.password, user.password_hash if user else DUMMY_PASSWORD_HASH)
    if user is None or not valid or user.role not in ("admin", "reader"):
        raise InvalidCredentialsError("email or password is wrong")

    auth = AuthContext(
        user_id=user.id,
        tenant_id=user.tenant_id,
        role=user.role,  # type: ignore[arg-type]  # checked just above
        email=user.email,
    )
    token, expires = create_access_token(auth)
    return LoginResponse(
        access_token=token, expires_at=expires, user=await _user_out(session, auth)
    )


@router.get("/me", response_model=UserOut)
async def me(auth: AuthDep, session: DbSession) -> UserOut:
    return await _user_out(session, auth)
