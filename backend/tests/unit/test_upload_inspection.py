"""The request-size limit (Giorno 26): a body over the limit is refused
from its headers, before the app reads it."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.core.limits import BodySizeLimit
from app.main import app

# -- request size ---------------------------------------------------------------


@pytest.fixture
def limited() -> Iterator[TestClient]:
    async def echo(request: Request) -> JSONResponse:
        return JSONResponse({"received": len(await request.body())})

    inner = Starlette(
        routes=[
            Route("/api/v1/documents", echo, methods=["POST"]),
            Route("/api/v1/chat", echo, methods=["POST"]),
        ]
    )
    with TestClient(BodySizeLimit(inner, max_upload_bytes=1000, max_request_bytes=100)) as client:
        yield client


def test_a_body_within_the_limit_reaches_the_app(limited: TestClient) -> None:
    response = limited.post("/api/v1/documents", content=b"x" * 1000)

    assert response.status_code == 200
    assert response.json() == {"received": 1000}


def test_an_upload_over_the_limit_is_refused_from_its_headers(limited: TestClient) -> None:
    response = limited.post("/api/v1/documents", content=b"x" * 1001)

    assert response.status_code == 413
    assert response.json()["code"] == "upload_too_large"


def test_other_endpoints_have_a_smaller_limit(limited: TestClient) -> None:
    assert limited.post("/api/v1/chat", content=b"x" * 100).status_code == 200
    assert limited.post("/api/v1/chat", content=b"x" * 101).status_code == 413


def test_a_body_without_a_declared_length_is_refused(limited: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        yield b"x" * 10

    response = limited.post("/api/v1/documents", content=chunks())

    assert response.status_code == 411
    assert response.json()["code"] == "length_required"


def test_the_app_refuses_a_200_mb_upload_before_reading_it() -> None:
    # Only the header claims 200 MB; nothing that large is ever sent. The
    # refusal comes before authentication, since nothing is read.
    response = TestClient(app).post(
        "/api/v1/documents",
        content=b"x",
        headers={"content-length": str(200 * 2**20), "content-type": "multipart/form-data"},
    )

    assert response.status_code == 413
