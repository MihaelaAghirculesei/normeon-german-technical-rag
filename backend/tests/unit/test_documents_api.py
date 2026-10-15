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

from app.api.deps import get_auth, get_embedder
from app.api.v1 import documents
from app.core.errors import EncryptedDocumentError
from app.db.models import Document
from app.main import app
from tests.unit.auth_helpers import authenticate

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
    """Answers the few queries these endpoints make; ingestion itself is
    stubbed out, so nothing else on AsyncSession is reached."""

    def __init__(self, document: Document | None, chunk_counts: list[tuple[UUID, str, int]]):
        self._document = document
        self._chunk_counts = chunk_counts
        self.statements: list[Any] = []

    async def get(self, model: type[Document], pk: UUID) -> Document | None:
        return self._document

    async def scalar(self, statement: Any) -> Document | None:
        self.statements.append(statement)
        return self._document

    async def scalars(self, statement: Any) -> list[Document]:
        self.statements.append(statement)
        return [] if self._document is None else [self._document]

    async def execute(self, statement: Any) -> list[tuple[UUID, str, int]]:
        self.statements.append(statement)
        return self._chunk_counts


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    def make(
        document: Document | None,
        created: bool = False,
        chunk_counts: list[tuple[UUID, str, int]] | None = None,
    ) -> tuple[TestClient, list[UUID], _FakeSession, list[dict[str, Any]]]:
        processed: list[UUID] = []
        create_calls: list[dict[str, Any]] = []
        session = _FakeSession(document, chunk_counts or [])

        async def fake_get_or_create(
            session: Any,
            tenant_id: UUID,
            filename: str,
            doc_type: str,
            content: bytes,
            **kwargs: Any,
        ) -> tuple[Document, bool]:
            assert document is not None
            create_calls.append({"filename": filename, "doc_type": doc_type, **kwargs})
            return document, created

        async def fake_process(
            tenant_id: UUID, document_id: UUID, content: bytes, embedder: Any
        ) -> None:
            assert tenant_id == TENANT_ID  # processing runs as the uploader's tenant
            processed.append(document_id)

        monkeypatch.setattr(documents, "get_or_create_document", fake_get_or_create)
        monkeypatch.setattr(documents, "_process_in_background", fake_process)
        # The fake bytes below are no real PDF: file inspection has its own
        # tests (test_upload_inspection.py); here it only has to pass.
        monkeypatch.setattr(documents, "inspect_upload", lambda *args: None)

        authenticate(app, session)
        app.dependency_overrides[get_embedder] = _FakeEmbedder
        return TestClient(app), processed, session, create_calls

    yield make
    app.dependency_overrides.clear()


def _upload(client: TestClient, filename: str = "FZV.pdf", **extra: str) -> Any:
    return client.post(
        "/api/v1/documents",
        data={"doc_type": "norm", **extra},
        files={"file": (filename, b"%PDF-1.4 fake", "application/octet-stream")},
    )


def test_uploading_a_new_document_returns_201_and_schedules_processing(
    client_factory: Any,
) -> None:
    client, processed, _, _ = client_factory(_document("pending"), created=True)

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
    client, processed, _, _ = client_factory(_document("ready"))

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
    client, processed, _, _ = client_factory(_document(status))

    response = _upload(client)

    assert response.status_code == 200
    assert response.json()["outcome"] == "in_progress"
    assert processed == []


def test_uploading_a_document_that_failed_before_retries_it(client_factory: Any) -> None:
    client, processed, _, _ = client_factory(_document("failed"))

    response = _upload(client)

    assert response.status_code == 200
    assert response.json()["outcome"] == "retrying"
    assert processed == [DOCUMENT_ID]


def test_upload_forwards_the_optional_version_fields(client_factory: Any) -> None:
    client, _, _, create_calls = client_factory(_document("pending"), created=True)

    _upload(client, version_label="v2.0", valid_from="2026-01-01")

    assert create_calls == [
        {
            "filename": "FZV.pdf",
            "doc_type": "norm",
            "version_label": "v2.0",
            "valid_from": date(2026, 1, 1),
            "valid_until": None,
        }
    ]


def test_upload_treats_a_blank_version_as_none(client_factory: Any) -> None:
    client, _, _, create_calls = client_factory(_document("pending"), created=True)

    _upload(client, version_label="")

    assert create_calls[0]["version_label"] is None


@pytest.mark.parametrize("filename", ["notes.txt", "scan.PNG", "README"])
def test_upload_rejects_a_file_type_no_parser_reads(client_factory: Any, filename: str) -> None:
    client, processed, _, create_calls = client_factory(_document("pending"), created=True)

    response = _upload(client, filename=filename)

    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_file_type"
    assert create_calls == []
    assert processed == []


def test_upload_accepts_an_uppercase_extension(client_factory: Any) -> None:
    client, _, _, _ = client_factory(_document("pending"), created=True)

    assert _upload(client, filename="LASTENHEFT.DOCX").status_code == 201


def test_a_file_refused_by_inspection_is_never_stored(
    client_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, processed, _, create_calls = client_factory(_document("pending"), created=True)

    def refuse(*args: Any) -> None:
        raise EncryptedDocumentError("the PDF is password-protected")

    monkeypatch.setattr(documents, "inspect_upload", refuse)

    response = _upload(client)

    assert response.status_code == 422
    assert response.json()["code"] == "encrypted_document"
    assert (create_calls, processed) == ([], [])


def test_getting_a_document_returns_its_current_status(client_factory: Any) -> None:
    client, _, _, _ = client_factory(_document("embedding"))

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
    client, _, _, _ = client_factory(
        document,
        chunk_counts=[(DOCUMENT_ID, "structural", 114), (DOCUMENT_ID, "fixed_500", 119)],
    )

    response = client.get("/api/v1/documents")

    assert response.status_code == 200
    [body] = response.json()
    assert body["document_id"] == str(DOCUMENT_ID)
    assert body["filename"] == "FZV.pdf"
    assert body["version_label"] == "v2.0"
    assert body["valid_from"] == "2026-01-01"
    assert body["valid_until"] is None
    assert body["status"] == "ready"
    assert body["chunk_counts"] == {"structural": 114, "fixed_500": 119}


def test_a_document_not_yet_parsed_has_no_chunk_counts(client_factory: Any) -> None:
    document = _document("pending")
    document.created_at = datetime(2026, 8, 26, 12, 0)
    client, _, _, _ = client_factory(document)

    [body] = client.get("/api/v1/documents").json()

    assert body["chunk_counts"] == {}


def test_listing_documents_is_scoped_to_the_tenant(client_factory: Any) -> None:
    client, _, session, _ = client_factory(None)

    response = client.get("/api/v1/documents")

    assert response.status_code == 200
    assert response.json() == []
    # Both the documents and the chunk-count query filter on the tenant.
    for statement in session.statements:
        compiled = statement.compile()
        assert "tenant_id = :tenant_id_1" in str(compiled)
        assert compiled.params["tenant_id_1"] == TENANT_ID


def test_documents_need_an_authenticated_caller(client_factory: Any) -> None:
    client, _, _, _ = client_factory(None)
    app.dependency_overrides.pop(get_auth)

    response = client.get("/api/v1/documents")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_a_reader_cannot_upload(client_factory: Any) -> None:
    client, processed, session, create_calls = client_factory(_document("pending"), created=True)
    authenticate(app, session, role="reader")

    response = _upload(client)

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"
    assert (create_calls, processed) == ([], [])


def test_the_status_read_is_confined_to_the_callers_tenant(client_factory: Any) -> None:
    client, _, session, _ = client_factory(_document("ready"))

    client.get(f"/api/v1/documents/{DOCUMENT_ID}")

    compiled = session.statements[-1].compile()
    assert compiled.params["tenant_id_1"] == TENANT_ID


def test_getting_an_unknown_document_returns_a_typed_404(client_factory: Any) -> None:
    client, _, _, _ = client_factory(None)

    response = client.get(f"/api/v1/documents/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["code"] == "document_not_found"
