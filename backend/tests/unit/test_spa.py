from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.spa import mount_spa


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    site = tmp_path / "browser"
    site.mkdir()
    (site / "index.html").write_text("<app-root></app-root>", encoding="utf-8")
    (site / "main-ABC123.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("not public", encoding="utf-8")

    app = FastAPI()

    @app.get("/api/v1/ping")
    async def ping() -> dict[str, str]:
        return {"pong": "yes"}

    mount_spa(app, site)
    return TestClient(app)


def test_a_bundle_is_served_and_cached_forever(client: TestClient) -> None:
    response = client.get("/main-ABC123.js")

    assert response.status_code == 200
    assert response.text == "console.log(1)"
    assert "immutable" in response.headers["cache-control"]


def test_the_root_and_a_client_side_route_get_the_app(client: TestClient) -> None:
    for path in ("/", "/fragen", "/dokumente/123"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.text == "<app-root></app-root>"
        assert response.headers["cache-control"] == "no-cache"


def test_api_routes_still_answer(client: TestClient) -> None:
    assert client.get("/api/v1/ping").json() == {"pong": "yes"}


def test_an_unknown_api_route_stays_a_json_404(client: TestClient) -> None:
    response = client.get("/api/v1/nope")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_files_outside_the_site_are_not_served(client: TestClient) -> None:
    response = client.get("/%2e%2e/secret.txt")

    assert "not public" not in response.text
