"""The one place the API's OpenAPI document is rendered -- so
`scripts/export_openapi.py` (which writes `backend/openapi.json`) and
`tests/unit/test_openapi.py` (which guards it against drift) agree on the
exact bytes. The frontend generates its typed client from that file, so a
contract change shows up as a stale-file test failure here and as a type
error in the frontend build, never as silent drift.
"""

from __future__ import annotations

import json

from app.main import app


def openapi_json() -> str:
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"
