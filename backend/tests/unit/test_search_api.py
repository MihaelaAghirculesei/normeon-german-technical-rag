"""Endpoint-level tests for POST /api/v1/search.

The DB session, the embedder, and the retrieval call itself are all
stubbed, so these check the HTTP contract -- request validation, response
shape, parameter forwarding -- without a database or a real model.
"""

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_embedder, get_session
from app.api.v1 import search
from app.domain.models import RetrievedChunk
from app.main import app

TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


class _FakeEmbedder:
    name = "fake"
    dim = 2

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0]


def _hit(score: float, section: str = "5.1.2") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="UN-R79.pdf",
        content="Die zulaessige Lenkkraft betraegt 300 N.",
        page_from=14,
        page_to=14,
        section_path=section,
        heading="Lenkkraft",
        score=score,
    )


@pytest.fixture
def client_with_retriever(monkeypatch: pytest.MonkeyPatch) -> Any:
    def make(hits: list[RetrievedChunk]) -> tuple[TestClient, list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []

        async def fake_vector_search(
            session: Any,
            embedder: Any,
            *,
            tenant_id: uuid.UUID,
            question: str,
            strategy: str | None,
            k: int | None,
            ef_search: int | None,
        ) -> list[RetrievedChunk]:
            calls.append(
                {
                    "tenant_id": tenant_id,
                    "question": question,
                    "strategy": strategy,
                    "k": k,
                    "ef_search": ef_search,
                }
            )
            return hits

        monkeypatch.setattr(search, "vector_search", fake_vector_search)
        app.dependency_overrides[get_session] = lambda: object()
        app.dependency_overrides[get_embedder] = _FakeEmbedder
        return TestClient(app), calls

    yield make
    app.dependency_overrides.clear()


def test_search_returns_hits_in_order_with_timing(client_with_retriever: Any) -> None:
    client, _ = client_with_retriever([_hit(0.91), _hit(0.72, section="5.1.3")])

    response = client.post(
        "/api/v1/search",
        json={"question": "Welche Lenkkraft ist zulaessig?", "tenant_id": str(TENANT_ID)},
    )

    assert response.status_code == 200
    body = response.json()
    assert [h["score"] for h in body["hits"]] == [0.91, 0.72]
    assert body["hits"][0]["filename"] == "UN-R79.pdf"
    assert body["hits"][0]["page_from"] == 14
    assert isinstance(body["took_ms"], (int, float)) and body["took_ms"] >= 0


def test_search_forwards_overrides_to_the_retriever(client_with_retriever: Any) -> None:
    client, calls = client_with_retriever([])

    client.post(
        "/api/v1/search",
        json={
            "question": "LH-3.2.1",
            "tenant_id": str(TENANT_ID),
            "strategy": "fixed_500",
            "k": 3,
            "ef_search": 80,
        },
    )

    assert calls == [
        {
            "tenant_id": TENANT_ID,
            "question": "LH-3.2.1",
            "strategy": "fixed_500",
            "k": 3,
            "ef_search": 80,
        }
    ]


def test_search_leaves_unset_params_as_none_for_the_service_to_default(
    client_with_retriever: Any,
) -> None:
    client, calls = client_with_retriever([])

    client.post(
        "/api/v1/search",
        json={"question": "Fahrzeugzulassung", "tenant_id": str(TENANT_ID)},
    )

    assert calls[0]["strategy"] is None
    assert calls[0]["k"] is None
    assert calls[0]["ef_search"] is None


def test_search_rejects_a_blank_question(client_with_retriever: Any) -> None:
    client, _ = client_with_retriever([])

    response = client.post(
        "/api/v1/search", json={"question": "", "tenant_id": str(TENANT_ID)}
    )

    assert response.status_code == 422


def test_search_rejects_k_above_the_cap(client_with_retriever: Any) -> None:
    client, _ = client_with_retriever([])

    response = client.post(
        "/api/v1/search",
        json={"question": "x", "tenant_id": str(TENANT_ID), "k": 500},
    )

    assert response.status_code == 422
