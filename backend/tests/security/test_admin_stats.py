"""The admin statistics (Giorno 27) against a real Postgres: admins only,
and only the caller's own tenant -- the numbers are confidential too."""

from fastapi.testclient import TestClient

from tests.security.conftest import B_QUESTION, login


def _stats(client: TestClient, email: str) -> dict[str, object]:
    response = client.get("/api/v1/admin/stats", headers=login(client, email))
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


def test_a_reader_gets_no_statistics(client: TestClient) -> None:
    response = client.get("/api/v1/admin/stats", headers=login(client, "ben@b.test"))

    assert response.status_code == 403


def test_statistics_count_only_the_callers_tenant(client: TestClient) -> None:
    before = _stats(client, "anna@a.test")
    assert before["latency_ms"] is None or before["queries"]

    # Anna (A) asks once; Ben (B) asks twice. Anna must see her one.
    client.post(
        "/api/v1/chat",
        json={"question": "Wie hoch ist der Oelpreis?"},
        headers=login(client, "anna@a.test"),
    )
    for _ in range(2):
        client.post(
            "/api/v1/chat", json={"question": B_QUESTION}, headers=login(client, "ben@b.test")
        )

    after = _stats(client, "anna@a.test")

    assert after["queries"] == int(before["queries"]) + 1  # type: ignore[call-overload]
    recent = after["recent"]
    assert isinstance(recent, list)
    assert [q["question"] for q in recent][0] == "Wie hoch ist der Oelpreis?"
    assert B_QUESTION not in {q["question"] for q in recent}
    # A has no documents, so its question was an abstention.
    assert after["abstention_rate"] == 1.0
    latency = after["latency_ms"]
    assert isinstance(latency, dict)
    assert set(latency) == {"retrieval_ms", "generation_ms", "total_ms"}
    assert latency["total_ms"]["p95"] >= latency["total_ms"]["p50"] >= 0
    documents = after["documents"]
    assert isinstance(documents, dict)
    assert documents == {"ready": 0, "processing": 0, "failed": 0, "chunks": 0}


def test_another_admin_sees_none_of_it(client: TestClient) -> None:
    client.post("/api/v1/chat", json={"question": B_QUESTION}, headers=login(client, "ben@b.test"))

    carla = _stats(client, "carla@c.test")

    assert carla["queries"] == 0
    assert carla["recent"] == []
    assert carla["latency_ms"] is None
    assert carla["abstention_rate"] is None
