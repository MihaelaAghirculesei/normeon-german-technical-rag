"""Regenerate backend/eval/dataset/schema.json from the EvalQuestion
Pydantic model (plan, Giorno 16). The model is the single source of
truth; this file is the checked-in artefact, and
tests/unit/test_eval_schema.py fails if the two drift.

    cd backend && .venv/Scripts/python scripts/gen_eval_schema.py
"""

import sys
from pathlib import Path

from app.eval.schema import question_schema_json

SCHEMA_PATH = Path(__file__).parents[1] / "eval" / "dataset" / "schema.json"


def main() -> int:
    SCHEMA_PATH.write_text(question_schema_json(), encoding="utf-8")
    print(f"wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
