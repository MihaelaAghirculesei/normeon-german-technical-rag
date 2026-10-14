"""Password hashing, access tokens and the startup guard against the
default JWT secret. Each token test changes exactly one thing about an
otherwise valid token and expects it to be refused."""

import base64
import json
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import Settings, insecure_settings, settings
from app.core.errors import NotAuthenticatedError
from app.core.security import (
    AuthContext,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

AUTH = AuthContext(
    user_id=uuid.uuid4(), tenant_id=uuid.uuid4(), role="admin", email="anna@demo.normeon.dev"
)


def _claims(**override: object) -> dict[str, object]:
    now = datetime.now(UTC)
    claims: dict[str, object] = {
        "sub": str(AUTH.user_id),
        "tenant_id": str(AUTH.tenant_id),
        "role": AUTH.role,
        "email": AUTH.email,
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "iss": settings.jwt_issuer,
    }
    claims.update(override)
    return {k: v for k, v in claims.items() if v is not None}


def _token(secret: str = settings.jwt_secret, **override: object) -> str:
    return jwt.encode(_claims(**override), secret, algorithm="HS256")


# --- passwords -------------------------------------------------------------


def test_a_password_verifies_against_its_own_hash_only() -> None:
    stored = hash_password("korrekt-pferd-batterie")

    assert verify_password("korrekt-pferd-batterie", stored)
    assert not verify_password("korrekt-pferd-batteriE", stored)


def test_hashes_are_salted() -> None:
    assert hash_password("same") != hash_password("same")


@pytest.mark.parametrize("stored", ["", "plain-text", "bcrypt$1$2$3$x$y", "scrypt$x$8$1$AA$AA"])
def test_a_malformed_hash_never_verifies(stored: str) -> None:
    assert not verify_password("anything", stored)


# --- tokens ----------------------------------------------------------------


def test_a_token_round_trips_the_auth_context() -> None:
    token, expires = create_access_token(AUTH)

    assert decode_access_token(token) == AUTH
    assert expires > datetime.now(UTC)


def test_an_expired_token_is_refused() -> None:
    past = datetime.now(UTC) - timedelta(hours=1)
    with pytest.raises(NotAuthenticatedError):
        decode_access_token(_token(iat=past - timedelta(hours=1), exp=past))


def test_a_token_signed_with_another_secret_is_refused() -> None:
    with pytest.raises(NotAuthenticatedError):
        decode_access_token(_token(secret="someone-elses-secret-that-is-long-enough"))


def test_an_unsigned_alg_none_token_is_refused() -> None:
    def part(data: dict[str, object]) -> str:
        raw = json.dumps(data, default=lambda d: int(d.timestamp())).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    unsigned = f"{part({'alg': 'none', 'typ': 'JWT'})}.{part(_claims())}."
    with pytest.raises(NotAuthenticatedError):
        decode_access_token(unsigned)


def test_a_token_from_another_issuer_is_refused() -> None:
    with pytest.raises(NotAuthenticatedError):
        decode_access_token(_token(iss="someone-else"))


@pytest.mark.parametrize("claim", ["tenant_id", "sub", "role", "exp"])
def test_a_token_missing_a_required_claim_is_refused(claim: str) -> None:
    with pytest.raises(NotAuthenticatedError):
        decode_access_token(_token(**{claim: None}))


def test_a_token_with_an_unknown_role_is_refused() -> None:
    with pytest.raises(NotAuthenticatedError):
        decode_access_token(_token(role="superadmin"))


def test_a_token_with_a_malformed_tenant_is_refused() -> None:
    with pytest.raises(NotAuthenticatedError):
        decode_access_token(_token(tenant_id="not-a-uuid"))


@pytest.mark.parametrize("garbage", ["", "abc", "a.b.c"])
def test_garbage_is_refused(garbage: str) -> None:
    with pytest.raises(NotAuthenticatedError):
        decode_access_token(garbage)


# --- startup guard ---------------------------------------------------------


def test_the_default_secret_is_fine_in_development() -> None:
    assert insecure_settings(Settings(environment="development")) == []


@pytest.mark.parametrize("secret", [None, "short"])
def test_production_refuses_a_default_or_short_secret(secret: str | None) -> None:
    s = (
        Settings(environment="production")
        if secret is None
        else Settings(environment="production", jwt_secret=secret)
    )

    assert insecure_settings(s) != []


def test_production_accepts_a_long_random_secret() -> None:
    s = Settings(environment="production", jwt_secret="x" * 48)

    assert insecure_settings(s) == []
