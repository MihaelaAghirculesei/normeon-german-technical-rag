"""Endpoint-level tests for the documents API.

The database session, the embedder, and the background processing step are
all overridden, so these exercise the HTTP contract -- status codes,
response shape, error mapping -- without a database or a real model.
"""

import uuid
from datetime import date, datetime
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

    async def scalars(self, statement: Any) -> list[Document]:
        self.statement = statement
        return [] if self._document is None else [self._document]


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    def make(
        document: Document | None, created: bool = False
    ) -> tuple[TestClient, list[UUID]]:
        processed: list[UUID] = []

        async def fake_get_or_create(
            session: Any, tenant_id: UUID, filename: str, doc_type: str, content: bytes
        ) -> tuple[Document, bool]:
            assert document is not None
            return document, created

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
    client, processed = client_factory(_document("pending"), created=True)

    response = _upload(client)

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == str(DOCUMENT_ID)
    assert body["status"] == "pending"
    assert body["outcome"] == "created"
    assert processed == [DOCUMENT_ID]


def test_uploading_an_already_ingested_document_returns_200_and_does_no_work(
    client_factory: Any,
) -> None:
    client, processed = client_factory(_document("ready"))

    response = _upload(client)

    # 201 Created would be a lie: nothing was created this time.
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "duplicate"
    assert body["status"] == "ready"
    assert processed == []


@pytest.mark.parametrize("status", ["pending", "parsing", "embedding"])
def test_uploading_a_document_still_being_processed_does_not_process_it_twice(
    client_factory: Any, status: str
) -> None:
    client, processed = client_factory(_document(status))

    response = _upload(client)

    assert response.status_code == 200
    assert response.json()["outcome"] == "in_progress"
    assert processed == []


def test_uploading_a_document_that_failed_before_retries_it(client_factory: Any) -> None:
    client, processed = client_factory(_document("failed"))

    response = _upload(client)

    assert response.status_code == 200
    assert response.json()["outcome"] == "retrying"
    assert processed == [DOCUMENT_ID]


def test_getting_a_document_returns_its_current_status(client_factory: Any) -> None:
    client, _ = client_factory(_document("embedding"))

    response = client.get(f"/api/v1/documents/{DOCUMENT_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "embedding"
    # outcome belongs to the upload response only.
    assert "outcome" not in body


def test_listing_documents_returns_summaries(client_factory: Any) -> None:
    document = _document("ready")
    document.version_label = "v2.0"
    document.valid_from = date(2026, 1, 1)
    document.created_at = datetime(2026, 8, 26, 12, 0)
    client, _ = client_factory(document)

    response = client.get("/api/v1/documents", params={"tenant_id": str(TENANT_ID)})

    assert response.status_code == 200
    [body] = response.json()
    assert body["document_id"] == str(DOCUMENT_ID)
    assert body["filename"] == "FZV.pdf"
    assert body["version_label"] == "v2.0"
    assert body["valid_from"] == "2026-01-01"
    assert body["valid_until"] is None
    assert body["status"] == "ready"


def test_listing_documents_is_scoped_to_the_tenant(client_factory: Any) -> None:
    client, _ = client_factory(None)
    session = app.dependency_overrides[get_session]()
    app.dependency_overrides[get_session] = lambda: session

    response = client.get("/api/v1/documents", params={"tenant_id": str(TENANT_ID)})

    assert response.status_code == 200
    assert response.json() == []
    compiled = session.statement.compile()
    assert "WHERE documents.tenant_id = :tenant_id_1" in str(compiled)
    assert compiled.params["tenant_id_1"] == TENANT_ID


def test_listing_documents_requires_a_tenant(client_factory: Any) -> None:
    client, _ = client_factory(None)

    assert client.get("/api/v1/documents").status_code == 422


def test_getting_an_unknown_document_returns_a_typed_404(client_factory: Any) -> None:
    client, _ = client_factory(None)

    response = client.get(f"/api/v1/documents/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["code"] == "document_not_found"
