"""The checked-in backend/eval/dataset/schema.json must stay in sync with
the EvalQuestion model it's generated from, and both the smoke question
set and the real 50-question set must validate against it.
"""

from collections import Counter
from pathlib import Path

from app.eval.dataset import DATASET_DIR, SMOKE_QUESTIONS, load_questions
from app.eval.models import CATEGORIES
from app.eval.schema import question_schema_json

SCHEMA_PATH = Path(__file__).parents[2] / "eval" / "dataset" / "schema.json"
QUESTIONS_PATH = DATASET_DIR / "questions.yaml"


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


def test_full_question_set_has_fifty_entries_ten_per_category() -> None:
    """Plan, Giorni 17-18 'Fatto quando': questions.yaml has 50 entries
    validated against the schema in CI."""
    questions = load_questions(QUESTIONS_PATH)

    assert len(questions) == 50
    assert Counter(q.category for q in questions) == {c: 10 for c in CATEGORIES}
    assert len({q.id for q in questions}) == 50
    assert sum(q.should_abstain for q in questions) == 10
