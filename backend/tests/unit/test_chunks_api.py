"""Endpoint-level tests for GET /api/v1/chunks/{id}/context. The service
is stubbed, so these pin the HTTP contract -- response shape, parameter
forwarding and validation, the typed 404 -- not the neighbour query
(that runs against a real Postgres in tests/integration/test_chunk_context.py).
"""

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import chunks
from app.core.errors import ChunkNotFoundError
from app.db.models import Chunk, Document
from app.main import app
from app.services.chunks import ChunkContext
from tests.unit.auth_helpers import authenticate

TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
DOCUMENT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


def _chunk(ordinal: int) -> Chunk:
    return Chunk(
        id=uuid.UUID(int=ordinal + 100),
        document_id=DOCUMENT_ID,
        tenant_id=TENANT_ID,
        strategy="structural",
        chunk_hash="0" * 64,
        ordinal=ordinal,
        page_from=40 + ordinal,
        page_to=40 + ordinal,
        section_path="§ 38 Lenkeinrichtung",
        heading=None,
        char_start=0,
        char_end=10,
        content=f"Absatz {ordinal}",
        content_norm=f"absatz {ordinal}",
        token_count=2,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    calls: list[dict[str, Any]] = []

    async def fake_get_chunk_context(session: Any, **kwargs: Any) -> ChunkContext:
        calls.append(kwargs)
        if kwargs["chunk_id"] != _chunk(2).id:
            raise ChunkNotFoundError("nope")
        document = Document(
            id=DOCUMENT_ID,
            tenant_id=TENANT_ID,
            content_hash="0" * 64,
            filename="StVZO.pdf",
            doc_type="norm",
            version_label=None,
        )
        return ChunkContext(
            document=document, chunk=_chunk(2), before=[_chunk(1)], after=[_chunk(3)]
        )

    monkeypatch.setattr(chunks, "get_chunk_context", fake_get_chunk_context)
    authenticate(app, object())
    yield TestClient(app), calls
    app.dependency_overrides.clear()


def test_context_returns_the_chunk_between_its_neighbours(client: Any) -> None:
    http, calls = client

    response = http.get(f"/api/v1/chunks/{_chunk(2).id}/context")

    assert response.status_code == 200
    body = response.json()
    assert body["document"]["filename"] == "StVZO.pdf"
    assert body["chunk"]["content"] == "Absatz 2"
    assert body["chunk"]["page_from"] == 42
    assert [c["content"] for c in body["before"]] == ["Absatz 1"]
    assert [c["content"] for c in body["after"]] == ["Absatz 3"]
    assert calls == [{"tenant_id": TENANT_ID, "chunk_id": _chunk(2).id, "window": 1}]


def test_context_forwards_the_window(client: Any) -> None:
    http, calls = client

    http.get(
        f"/api/v1/chunks/{_chunk(2).id}/context",
        params={"window": 3},
    )

    assert calls[0]["window"] == 3


@pytest.mark.parametrize("window", [-1, 4])
def test_context_rejects_a_window_out_of_range(client: Any, window: int) -> None:
    http, _ = client

    response = http.get(
        f"/api/v1/chunks/{_chunk(2).id}/context",
        params={"window": window},
    )

    assert response.status_code == 422


def test_context_of_an_unknown_chunk_is_a_typed_404(client: Any) -> None:
    http, _ = client

    response = http.get(f"/api/v1/chunks/{uuid.uuid4()}/context")

    assert response.status_code == 404
    assert response.json()["code"] == "chunk_not_found"
