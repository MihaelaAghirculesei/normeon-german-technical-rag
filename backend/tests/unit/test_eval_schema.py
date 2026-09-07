"""The checked-in backend/eval/dataset/schema.json must stay in sync with
the EvalQuestion model it's generated from, and the smoke question set
must validate against it.
"""

from pathlib import Path

from app.eval.dataset import SMOKE_QUESTIONS, load_questions
from app.eval.schema import question_schema_json

SCHEMA_PATH = Path(__file__).parents[2] / "eval" / "dataset" / "schema.json"


def test_schema_json_is_in_sync_with_the_model() -> None:
    assert SCHEMA_PATH.read_text(encoding="utf-8") == question_schema_json(), (
        "backend/eval/dataset/schema.json is stale -- run "
        "`.venv/Scripts/python scripts/gen_eval_schema.py`"
    )


def test_smoke_question_set_validates() -> None:
    questions = load_questions(SMOKE_QUESTIONS)

    assert len(questions) == 3
    assert {q.id for q in questions} == {"Q001", "Q002", "Q003"}
    assert any(q.should_abstain for q in questions)
