"""Password hashing and access tokens (Giorno 25).

Passwords: scrypt from the standard library -- memory-hard, no extra
dependency -- stored as a self-describing string so the cost parameters
can be raised later without invalidating existing hashes.

Tokens: HS256 JWTs carrying the user id, tenant id and role. The tenant in
the token is the only source of tenant identity for a request; nothing a
client sends in a body or query string can choose a tenant.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID

import jwt

from app.core.config import settings
from app.core.errors import NotAuthenticatedError

Role = Literal["admin", "reader"]
ROLES: frozenset[str] = frozenset({"admin", "reader"})

# OWASP's scrypt baseline (N=2^17, r=8, p=1) needs 128 MiB per hash; N=2^14
# keeps a login at ~16 MiB / tens of ms on this kind of hardware. The
# parameters are stored with each hash, so raising them is a config change.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_KEY_BYTES = 32
_ALGORITHM = "HS256"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def hash_password(password: str) -> str:
    salt = os.urandom(_SALT_BYTES)
    key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_KEY_BYTES,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(key)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, key = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(key)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=base64.b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# A fixed hash to verify against when the email is unknown, so a login for
# a non-existent account costs the same time as one with a wrong password.
DUMMY_PASSWORD_HASH = hash_password("not-a-real-password")


@dataclass(frozen=True, slots=True)
class AuthContext:
    """Who is calling: decoded from the access token, trusted from then on."""

    user_id: UUID
    tenant_id: UUID
    role: Role
    email: str


def create_access_token(auth: AuthContext, *, now: datetime | None = None) -> tuple[str, datetime]:
    issued = now or datetime.now(UTC)
    expires = issued + timedelta(minutes=settings.jwt_ttl_minutes)
    claims = {
        "sub": str(auth.user_id),
        "tenant_id": str(auth.tenant_id),
        "role": auth.role,
        "email": auth.email,
        "iat": issued,
        "exp": expires,
        "iss": settings.jwt_issuer,
    }
    token = jwt.encode(claims, settings.jwt_secret, algorithm=_ALGORITHM)
    return token, expires


def decode_access_token(token: str) -> AuthContext:
    """Verify signature, expiry and issuer, and require every claim we
    rely on. Any failure is a plain `NotAuthenticatedError`."""
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            # Pinned: never let the token's own header pick the algorithm.
            algorithms=[_ALGORITHM],
            issuer=settings.jwt_issuer,
            options={"require": ["sub", "tenant_id", "role", "exp", "iat", "iss"]},
        )
        role = claims["role"]
        if role not in ROLES:
            raise NotAuthenticatedError("unknown role")
        return AuthContext(
            user_id=UUID(claims["sub"]),
            tenant_id=UUID(claims["tenant_id"]),
            role=role,
            email=str(claims.get("email", "")),
        )
    except (jwt.PyJWTError, ValueError, KeyError) as exc:
        raise NotAuthenticatedError("invalid or expired token") from exc
