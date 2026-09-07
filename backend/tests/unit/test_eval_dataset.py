"""Unit tests for app/eval/dataset.load_questions."""

from pathlib import Path

import pytest

from app.eval.dataset import load_questions

_VALID = """
- id: Q001
  category: requirement_lookup
  question: "Welche Lenkkraft ist zulaessig?"
  expected_answer_points: ["300 N"]
  gold_sources:
    - { document: "UN-R79.pdf", section: "5.1.2", page: 14 }
  should_abstain: false
  difficulty: medium
- id: Q002
  category: unanswerable
  question: "Nicht im Korpus."
  should_abstain: true
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_and_validates_a_good_file(tmp_path: Path) -> None:
    questions = load_questions(_write(tmp_path, _VALID))

    assert [q.id for q in questions] == ["Q001", "Q002"]
    assert questions[0].expected_answer_points == ["300 N"]
    assert questions[0].gold_sources[0].page == 14
    assert questions[1].should_abstain is True
    assert questions[1].difficulty == "medium"  # default


def test_a_bad_id_pattern_raises_with_the_question_number(tmp_path: Path) -> None:
    bad = "- id: XYZ\n  category: code_lookup\n  question: q\n"

    with pytest.raises(ValueError, match="question #1 is invalid"):
        load_questions(_write(tmp_path, bad))


def test_an_unknown_category_raises(tmp_path: Path) -> None:
    bad = "- id: Q009\n  category: made_up\n  question: q\n"

    with pytest.raises(ValueError, match="question #1 is invalid"):
        load_questions(_write(tmp_path, bad))


def test_duplicate_ids_raise(tmp_path: Path) -> None:
    dup = (
        "- id: Q001\n  category: code_lookup\n  question: a\n"
        "- id: Q001\n  category: code_lookup\n  question: b\n"
    )

    with pytest.raises(ValueError, match="duplicate question id 'Q001'"):
        load_questions(_write(tmp_path, dup))


def test_a_non_list_top_level_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected a top-level YAML list"):
        load_questions(_write(tmp_path, "id: Q001\ncategory: code_lookup\nquestion: q\n"))
