"""The one place the `EvalQuestion` JSON Schema is rendered -- so
`scripts/gen_eval_schema.py` (which writes it) and
`tests/unit/test_eval_schema.py` (which guards it against drift) agree
on the exact bytes.
"""

from __future__ import annotations

import json

from app.eval.models import EvalQuestion


def question_schema_json() -> str:
    return json.dumps(EvalQuestion.model_json_schema(), indent=2, sort_keys=True) + "\n"
