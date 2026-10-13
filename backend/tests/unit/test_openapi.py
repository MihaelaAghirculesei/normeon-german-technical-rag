"""The checked-in backend/openapi.json must stay in sync with the app it's
generated from: the frontend's typed client is built from that file, so a
stale copy would let the two sides drift apart unnoticed.
"""

from pathlib import Path

from app.openapi import openapi_json

OPENAPI_PATH = Path(__file__).parents[2] / "openapi.json"


def test_openapi_json_is_in_sync_with_the_app() -> None:
    assert OPENAPI_PATH.read_text(encoding="utf-8") == openapi_json(), (
        "backend/openapi.json is stale -- run `.venv/Scripts/python scripts/export_openapi.py`"
    )
