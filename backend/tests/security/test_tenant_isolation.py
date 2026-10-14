"""Tenant isolation through the HTTP API (Giorno 25).

The headline case: a user of tenant A asks a question whose answer exists
only in tenant B's documents and gets NICHT_GEFUNDEN -- not a 403, because
even the existence of B's document is confidential. Every test that shows
A seeing nothing pairs it with B seeing the thing, so a broken index or an
empty database cannot make the isolation test pass by accident.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.security.conftest import B_QUESTION, SecurityDb, build_pdf, login


def _chat(client: TestClient, headers: dict[str, str], question: str) -> dict[str, object]:
    response = client.post("/api/v1/chat", json={"question": question}, headers=headers)
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


def test_a_question_answerable_only_in_another_tenant_gets_nicht_gefunden(
    client: TestClient,
) -> None:
    answer_for_b = _chat(client, login(client, "ben@b.test"), B_QUESTION)
    assert answer_for_b["answer"] != "NICHT_GEFUNDEN"
    assert {s["filename"] for s in answer_for_b["sources"]} == {"Lastenheft-B.pdf"}  # type: ignore[union-attr]

    answer_for_a = _chat(client, login(client, "anna@a.test"), B_QUESTION)

    assert answer_for_a["answer"] == "NICHT_GEFUNDEN"
    assert answer_for_a["sources"] == []
    assert answer_for_a["citations"] == []


def test_a_tenant_id_smuggled_into_the_body_changes_nothing(
    client: TestClient, security_db: SecurityDb
) -> None:
    response = client.post(
        "/api/v1/chat",
        json={"question": B_QUESTION, "tenant_id": "bbbbbbbb-0000-0000-0000-000000000000"},
        headers=login(client, "anna@a.test"),
    )

    assert response.json()["answer"] == "NICHT_GEFUNDEN"


def test_the_streaming_answer_is_isolated_too(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"question": B_QUESTION},
        headers=login(client, "anna@a.test"),
    ) as response:
        body = "".join(response.iter_text())

    frames = {
        block.split("\n")[0].removeprefix("event: "): json.loads(block.split("\n")[1][6:])
        for block in body.strip().split("\n\n")
        if block.startswith("event: ") and block.split("\n")[0] != "event: stage"
    }
    assert frames["sources"] == {"sources": []}
    assert frames["token"] == {"text": "NICHT_GEFUNDEN"}


def test_another_tenants_document_does_not_exist_for_the_caller(
    client: TestClient, security_db: SecurityDb
) -> None:
    ben, anna = login(client, "ben@b.test"), login(client, "anna@a.test")
    document_url = f"/api/v1/documents/{security_db.b_document_id}"
    b_chunk_id = _chat(client, ben, B_QUESTION)["sources"][0]["chunk_id"]  # type: ignore[index]
    chunk_url = f"/api/v1/chunks/{b_chunk_id}/context"

    # Control: B sees its own document and chunk.
    assert [d["filename"] for d in client.get("/api/v1/documents", headers=ben).json()] == [
        "Lastenheft-B.pdf"
    ]
    assert client.get(document_url, headers=ben).status_code == 200
    assert client.get(chunk_url, headers=ben).status_code == 200

    # A: not in the list, and a direct id is a plain "not found", not 403.
    assert client.get("/api/v1/documents", headers=anna).json() == []
    assert client.get(document_url, headers=anna).status_code == 404
    assert client.get(chunk_url, headers=anna).status_code == 404
    search = client.post(
        "/api/v1/search", json={"question": B_QUESTION, "mode": "hybrid"}, headers=anna
    )
    assert search.status_code == 200
    assert search.json()["hits"] == []


def test_an_upload_is_ingested_into_the_uploaders_tenant_only(
    client: TestClient, tmp_path: Path
) -> None:
    carla, anna = login(client, "carla@c.test"), login(client, "anna@a.test")
    pdf = build_pdf(tmp_path / "c.pdf", [("1 Kennzeichnung", "Typschild und Kennzeichnung. " * 30)])

    response = client.post(
        "/api/v1/documents",
        data={"doc_type": "norm", "tenant_id": "aaaaaaaa-0000-0000-0000-000000000000"},
        files={"file": ("Norm-C.pdf", pdf, "application/pdf")},
        headers=carla,
    )

    assert response.status_code == 201, response.text
    # TestClient runs the background ingestion before returning: it parsed,
    # chunked and embedded under RLS, as tenant C.
    [listed] = client.get("/api/v1/documents", headers=carla).json()
    assert listed["status"] == "ready"
    assert listed["chunk_counts"]["structural"] > 0
    assert client.get("/api/v1/documents", headers=anna).json() == []


def test_a_reader_cannot_upload(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/v1/documents",
        data={"doc_type": "norm"},
        files={"file": ("x.pdf", build_pdf(tmp_path / "x.pdf", [("1 X", "Text. " * 20)]))},
        headers=login(client, "ben@b.test"),
    )

    assert response.status_code == 403


def test_every_tenant_endpoint_refuses_a_missing_or_forged_token(
    client: TestClient, security_db: SecurityDb
) -> None:
    forged = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.c2lnbmF0dXJl"}
    requests = [
        ("GET", "/api/v1/documents", None),
        ("GET", f"/api/v1/documents/{security_db.b_document_id}", None),
        ("POST", "/api/v1/chat", {"question": "x"}),
        ("POST", "/api/v1/chat/stream", {"question": "x"}),
        ("POST", "/api/v1/search", {"question": "x"}),
        ("GET", f"/api/v1/chunks/{security_db.b_document_id}/context", None),
        ("GET", "/api/v1/auth/me", None),
    ]
    for method, url, body in requests:
        for headers in (None, forged):
            response = client.request(method, url, json=body, headers=headers)
            assert response.status_code == 401, (method, url, headers)
