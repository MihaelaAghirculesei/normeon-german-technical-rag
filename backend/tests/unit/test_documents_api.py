"""Endpoint-level tests for the documents API.

The database session, the embedder, and the background processing step are
all overridden, so these exercise the HTTP contract -- status codes,
response shape, error mapping -- without a database or a real model.
"""

import uuid
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_embedder
from app.api.v1 import documents
from app.db.models import Document
from app.db.session import get_session
from app.main import app

TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
DOCUMENT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


class _FakeEmbedder:
    name = "fake"
    dim = 2

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0]


def _document(status: str) -> Document:
    return Document(
        id=DOCUMENT_ID,
        tenant_id=TENANT_ID,
        content_hash="0" * 64,
        filename="FZV.pdf",
        doc_type="norm",
        status=status,
    )


class _FakeSession:
    """Only `get` is exercised by these endpoints; ingestion itself is
    stubbed out, so nothing else on AsyncSession is reached."""

    def __init__(self, document: Document | None) -> None:
        self._document = document

    async def get(self, model: type[Document], pk: UUID) -> Document | None:
        return self._document


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    def make(
        document: Document | None, already_ingested: bool = False
    ) -> tuple[TestClient, list[UUID]]:
        processed: list[UUID] = []

        async def fake_get_or_create(
            session: Any, tenant_id: UUID, filename: str, doc_type: str, content: bytes
        ) -> tuple[Document, bool]:
            assert document is not None
            return document, already_ingested

        async def fake_process(document_id: UUID, content: bytes, embedder: Any) -> None:
            processed.append(document_id)

        monkeypatch.setattr(documents, "get_or_create_document", fake_get_or_create)
        monkeypatch.setattr(documents, "_process_in_background", fake_process)

        app.dependency_overrides[get_session] = lambda: _FakeSession(document)
        app.dependency_overrides[get_embedder] = _FakeEmbedder
        return TestClient(app), processed

    yield make
    app.dependency_overrides.clear()


def _upload(client: TestClient) -> Any:
    return client.post(
        "/api/v1/documents",
        data={"tenant_id": str(TENANT_ID), "doc_type": "norm"},
        files={"file": ("FZV.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )


def test_uploading_a_new_document_returns_201_and_schedules_processing(
    client_factory: Any,
) -> None:
    client, processed = client_factory(_document("pending"), already_ingested=False)

    response = _upload(client)

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == str(DOCUMENT_ID)
    assert body["status"] == "pending"
    assert body["already_ingested"] is False
    assert processed == [DOCUMENT_ID]


def test_uploading_an_already_ingested_document_returns_200_and_does_no_work(
    client_factory: Any,
) -> None:
    client, processed = client_factory(_document("ready"), already_ingested=True)

    response = _upload(client)

    # 201 Created would be a lie: nothing was created this time.
    assert response.status_code == 200
    body = response.json()
    assert body["already_ingested"] is True
    assert body["status"] == "ready"
    assert processed == []


def test_getting_a_document_returns_its_current_status(client_factory: Any) -> None:
    client, _ = client_factory(_document("embedding"))

    response = client.get(f"/api/v1/documents/{DOCUMENT_ID}")

    assert response.status_code == 200
    assert response.json()["status"] == "embedding"


def test_getting_an_unknown_document_returns_a_typed_404(client_factory: Any) -> None:
    client, _ = client_factory(None)

    response = client.get(f"/api/v1/documents/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["code"] == "document_not_found"
