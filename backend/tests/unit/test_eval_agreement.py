"""Unit tests for app/eval/agreement: human-review sampling, the
agreement metrics (checked against hand-worked/known values), and the
human-review worksheet writer."""

import csv
from pathlib import Path

import pytest

from app.eval.agreement import (
    cohens_kappa,
    percent_agreement,
    select_human_review_sample,
    write_human_review_worksheet,
)
from app.eval.models import CATEGORIES, EvalQuestion
from app.eval.scoring import JudgeError, JudgeResult, Layer1Result, ScoredQuestionRun


def _question(id_: str, category: str) -> EvalQuestion:
    return EvalQuestion.model_validate(
        {"id": id_, "category": category, "question": f"Frage {id_}?"}
    )


def test_select_human_review_sample_takes_three_per_category() -> None:
    questions = [
        _question(f"Q{i:03d}", category)
        for category in CATEGORIES
        for i in range(1, 6)  # 5 questions per category, 25 total
    ]

    sample = select_human_review_sample(questions)

    assert len(sample) == 15
    counts = {c: sum(1 for q in sample if q.category == c) for c in CATEGORIES}
    assert all(count == 3 for count in counts.values())


def test_select_human_review_sample_takes_the_first_ones_per_category() -> None:
    questions = [_question(f"Q{i:03d}", "code_lookup") for i in range(1, 11)]

    sample = select_human_review_sample(questions)

    assert [q.id for q in sample] == ["Q001", "Q002", "Q003"]


def test_percent_agreement_full_match() -> None:
    assert percent_agreement([2, 1, 0, 2], [2, 1, 0, 2]) == 1.0


def test_percent_agreement_partial_match() -> None:
    assert percent_agreement([2, 1, 0, 2], [2, 1, 1, 2]) == 0.75


def test_percent_agreement_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        percent_agreement([1, 2], [1])


def test_percent_agreement_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percent_agreement([], [])


def test_cohens_kappa_perfect_agreement_is_one() -> None:
    assert cohens_kappa([2, 1, 0, 2, 1], [2, 1, 0, 2, 1]) == 1.0


def test_cohens_kappa_all_same_single_label_both_raters() -> None:
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_cohens_kappa_known_confusion_matrix() -> None:
    # Two raters, 50 items, yes/no: 20 yes/yes, 5 yes/no, 10 no/yes, 15
    # no/no. po = (20+15)/50 = 0.7; p(A=yes)=p(A=no)=0.5,
    # p(B=yes)=30/50=0.6, p(B=no)=0.4 -> pe = 0.5*0.6 + 0.5*0.4 = 0.5.
    # kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4, hand-verified.
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.4, abs=1e-9)


def test_cohens_kappa_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa([1, 2], [1])


def test_cohens_kappa_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        cohens_kappa([], [])


def test_write_human_review_worksheet_has_fifteen_rows_and_a_blank_human_score_column(
    tmp_path: Path,
) -> None:
    questions = [
        _question(f"Q{i:03d}", category) for category in CATEGORIES for i in range(1, 6)
    ]
    scored = [
        ScoredQuestionRun(
            question_id=q.id,
            category=q.category,
            answer=f"Antwort auf {q.id}",
            layer1=Layer1Result(points_found=[], points_missing=[]),
            layer2=JudgeResult(score=2, rationale="passt"),
        )
        for q in questions
    ]

    path = write_human_review_worksheet(questions, scored, tmp_path / "review.csv")

    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    assert len(rows) == 15
    assert all(row["human_score"] == "" for row in rows)
    assert rows[0]["question_id"] == "Q001"
    assert rows[0]["answer"] == "Antwort auf Q001"
    assert rows[0]["judge_score"] == "2"


def test_write_human_review_worksheet_blanks_a_missing_scored_run(tmp_path: Path) -> None:
    questions = [_question(f"Q{i:03d}", "code_lookup") for i in range(1, 4)]

    path = write_human_review_worksheet(questions, [], tmp_path / "review.csv")

    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    assert all(row["answer"] == "" and row["judge_score"] == "" for row in rows)


def test_write_human_review_worksheet_blanks_judge_columns_on_a_judge_error(
    tmp_path: Path,
) -> None:
    questions = [_question("Q001", "code_lookup")]
    scored = [
        ScoredQuestionRun(
            question_id="Q001",
            category="code_lookup",
            answer="irgendeine Antwort",
            layer1=Layer1Result(points_found=[], points_missing=[]),
            layer2=JudgeError(raw_response="kein JSON", error="boom"),
        )
    ]

    path = write_human_review_worksheet(questions, scored, tmp_path / "review.csv")

    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    assert rows[0]["judge_score"] == ""
    assert rows[0]["judge_rationale"] == ""
