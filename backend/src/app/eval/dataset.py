"""Load and validate an evaluation question set (plan, Giorno 16).

A YAML list of question mappings in, a list of validated `EvalQuestion`
out. Validation is just Pydantic against `EvalQuestion` -- the same
model `backend/eval/dataset/schema.json` is generated from -- so a
malformed question fails here with a precise message rather than
surfacing as a confusing error deep in a run.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from app.eval.models import EvalQuestion

# backend/eval/dataset/ -- the plan's layout; not under src/app.
DATASET_DIR = Path(__file__).parents[3] / "eval" / "dataset"
SMOKE_QUESTIONS = DATASET_DIR / "questions.smoke.yaml"


def load_questions(path: Path) -> list[EvalQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: expected a top-level YAML list, got {type(raw).__name__}")

    questions: list[EvalQuestion] = []
    ids: set[str] = set()
    for i, entry in enumerate(raw):
        try:
            question = EvalQuestion.model_validate(entry)
        except ValidationError as exc:
            raise ValueError(f"{path.name}: question #{i + 1} is invalid:\n{exc}") from exc
        if question.id in ids:
            raise ValueError(f"{path.name}: duplicate question id {question.id!r}")
        ids.add(question.id)
        questions.append(question)
    return questions
