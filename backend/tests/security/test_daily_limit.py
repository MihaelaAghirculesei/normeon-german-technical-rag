"""The daily question cap (Giorno 29) against a real Postgres: it counts
every tenant's questions, although each tenant's own session sees only its
rows under row-level security -- one tenant cannot hide spend from it."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.services.budget import questions_today
from tests.security.conftest import B_QUESTION, SecurityDb, login


def _asked_today(security_db: SecurityDb) -> int:
    async def count() -> int:
        async with security_db.factory()() as session:
            return await questions_today(session)

    return asyncio.run(count())


def test_the_cap_counts_every_tenant_and_then_refuses(
    client: TestClient, security_db: SecurityDb, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(deps.settings, "daily_question_limit", _asked_today(security_db) + 2)
    question = {"question": B_QUESTION}

    for email in ("ben@b.test", "anna@a.test"):
        answered = client.post("/api/v1/chat", json=question, headers=login(client, email))
        assert answered.is_success, answered.text

    # Carla (tenant C) has asked nothing, but the day's budget is spent.
    refused = client.post("/api/v1/chat", json=question, headers=login(client, "carla@c.test"))
    streamed = client.post(
        "/api/v1/chat/stream", json=question, headers=login(client, "carla@c.test")
    )

    assert refused.status_code == 429
    assert refused.json()["code"] == "daily_limit_reached"
    assert int(refused.headers["retry-after"]) > 0
    assert streamed.status_code == 429
