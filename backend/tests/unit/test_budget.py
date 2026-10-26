from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.core.errors import DailyLimitReachedError
from app.main import app
from app.services.budget import ensure_daily_budget


class _CountingSession:
    """Answers the one count query `ensure_daily_budget` runs."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.queries = 0

    async def scalar(self, statement: Any) -> int:
        self.queries += 1
        return self.count


async def test_no_cap_means_no_query() -> None:
    session = _CountingSession(10_000)

    await ensure_daily_budget(session, None)  # type: ignore[arg-type]
    await ensure_daily_budget(session, 0)  # type: ignore[arg-type]

    assert session.queries == 0


async def test_under_the_cap_answers() -> None:
    await ensure_daily_budget(_CountingSession(199), 200)  # type: ignore[arg-type]


async def test_at_the_cap_refuses_until_midnight_utc() -> None:
    now = datetime(2026, 10, 26, 22, 30, tzinfo=UTC)

    with pytest.raises(DailyLimitReachedError) as refused:
        await ensure_daily_budget(_CountingSession(200), 200, now)  # type: ignore[arg-type]

    assert refused.value.status_code == 429
    assert refused.value.retry_after_s == 90 * 60


def test_the_refusal_reaches_the_client_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def spent() -> None:
        raise DailyLimitReachedError("spent", retry_after_s=120)

    app.dependency_overrides[deps.enforce_daily_question_limit] = spent
    try:
        response = TestClient(app).post("/api/v1/chat", json={"question": "Bremse?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 429
    assert response.json()["code"] == "daily_limit_reached"
    assert response.headers["retry-after"] == "120"
