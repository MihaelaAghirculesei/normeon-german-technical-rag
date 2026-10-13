"""Regenerate backend/openapi.json from the FastAPI app (plan, Giorno 21).
The app is the single source of truth; this file is the checked-in
contract the frontend generates its typed client from, and
tests/unit/test_openapi.py fails if the two drift.

    cd backend && .venv/Scripts/python scripts/export_openapi.py
"""

import sys
from pathlib import Path

from app.openapi import openapi_json

OPENAPI_PATH = Path(__file__).parents[1] / "openapi.json"


def main() -> int:
    OPENAPI_PATH.write_text(openapi_json(), encoding="utf-8", newline="\n")
    print(f"wrote {OPENAPI_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
