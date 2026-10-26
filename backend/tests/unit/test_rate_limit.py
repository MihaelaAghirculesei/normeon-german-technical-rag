import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core import rate_limit
from app.core.rate_limit import RateLimit, Rule


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _client(clock: _Clock, *, limit: int = 2, window_s: float = 60) -> TestClient:
    app = FastAPI()

    @app.post("/api/v1/chat")
    async def chat() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/api/v1/auth/login")
    async def login() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/v1/documents")
    async def documents() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(
        RateLimit,
        rules=[
            Rule("questions", "POST", frozenset({"/api/v1/chat"}), limit, window_s),
            Rule("login", "POST", frozenset({"/api/v1/auth/login"}), limit, window_s),
        ],
        clock=clock,
    )
    return TestClient(app)


def test_requests_over_the_limit_get_429_with_retry_after() -> None:
    clock = _Clock()
    client = _client(clock)

    assert client.post("/api/v1/chat").status_code == 200
    clock.now += 10
    assert client.post("/api/v1/chat").status_code == 200
    refused = client.post("/api/v1/chat")

    assert refused.status_code == 429
    assert refused.json()["code"] == "rate_limited"
    assert refused.headers["retry-after"] == "50"  # the first hit leaves the window at +60


def test_the_window_slides() -> None:
    clock = _Clock()
    client = _client(clock)
    client.post("/api/v1/chat")
    client.post("/api/v1/chat")

    clock.now += 60
    assert client.post("/api/v1/chat").status_code == 200


def test_rules_count_separately() -> None:
    clock = _Clock()
    client = _client(clock)
    client.post("/api/v1/chat")
    client.post("/api/v1/chat")

    assert client.post("/api/v1/chat").status_code == 429
    assert client.post("/api/v1/auth/login").status_code == 200


def test_paths_and_methods_without_a_rule_are_never_limited() -> None:
    clock = _Clock()
    client = _client(clock, limit=1)

    assert all(client.get("/api/v1/documents").status_code == 200 for _ in range(5))


def test_a_rule_with_limit_zero_is_off() -> None:
    clock = _Clock()
    client = _client(clock, limit=0)

    assert all(client.post("/api/v1/chat").status_code == 200 for _ in range(5))


async def test_each_client_ip_has_its_own_budget() -> None:
    # TestClient always sends from one address, so drive the middleware
    # directly with a scope per client.
    clock = _Clock()
    seen: list[int] = []

    async def app(scope, receive, send):  # type: ignore[no-untyped-def]
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    limiter = RateLimit(
        app, rules=[Rule("questions", "POST", frozenset({"/q"}), 1, 60)], clock=clock
    )

    async def call(ip: str) -> None:
        scope = {"type": "http", "method": "POST", "path": "/q", "client": (ip, 1), "headers": []}

        async def receive():  # type: ignore[no-untyped-def]
            return {"type": "http.request", "body": b""}

        async def send(message):  # type: ignore[no-untyped-def]
            if message["type"] == "http.response.start":
                seen.append(message["status"])

        await limiter(scope, receive, send)

    await call("1.1.1.1")
    await call("2.2.2.2")
    await call("1.1.1.1")

    assert seen == [200, 200, 429]


def test_clients_gone_quiet_are_forgotten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limit, "_SWEEP_AT", 3)
    clock = _Clock()

    async def app(scope, receive, send):  # type: ignore[no-untyped-def]
        pass

    limiter = RateLimit(app, rules=[Rule("q", "POST", frozenset({"/q"}), 5, 60)], clock=clock)
    for ip in ("a", "b", "c", "d"):
        limiter._retry_after(limiter.rules[0], ip)

    clock.now += 61
    limiter._retry_after(limiter.rules[0], "e")

    assert set(limiter._hits) == {("q", "e")}
