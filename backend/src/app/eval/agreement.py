"""Human-vs-judge agreement (plan, Giorno 18): "the step almost nobody
does" -- score a subset of answers by hand yourself, then measure how
much your judgments and the LLM-judge's agree, so Layer 2's numbers can
be trusted with eyes open rather than taken on faith.

Pure: no I/O, no LLM calls. `select_human_review_sample` only decides
*which* questions a human should score by hand; scoring them is a human
act this module doesn't automate.
"""

from __future__ import annotations

import csv
from pathlib import Path

from app.eval.models import CATEGORIES, EvalQuestion
from app.eval.scoring import JudgeResult, ScoredQuestionRun

HUMAN_REVIEW_SAMPLE_SIZE = 15
_PER_CATEGORY = HUMAN_REVIEW_SAMPLE_SIZE // len(CATEGORIES)
_WORKSHEET_HEADER = [
    "question_id",
    "category",
    "question",
    "answer",
    "expected_answer_points",
    "gold_sources",
    "judge_score",
    "judge_rationale",
    "human_score",
]


def select_human_review_sample(questions: list[EvalQuestion]) -> list[EvalQuestion]:
    """The first `_PER_CATEGORY` questions (by their position in the
    input, i.e. by id since questions.yaml is authored in id order) from
    each of the 5 categories -- deterministic and reproducible, and
    guarantees every category is represented rather than an arbitrary
    top-N slice that could land entirely inside one or two categories."""
    selected: list[EvalQuestion] = []
    for category in CATEGORIES:
        in_category = [q for q in questions if q.category == category]
        selected.extend(in_category[:_PER_CATEGORY])
    return selected


def percent_agreement(a: list[int], b: list[int]) -> float:
    if len(a) != len(b):
        raise ValueError(f"score lists must be the same length: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("need at least one scored pair")
    matches = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    return matches / len(a)


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Chance-corrected agreement. `po` is the observed agreement rate
    (same as `percent_agreement`); `pe` is the agreement rate expected if
    both raters assigned scores independently at their own observed
    label frequencies. `kappa = (po - pe) / (1 - pe)`."""
    if len(a) != len(b):
        raise ValueError(f"score lists must be the same length: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        raise ValueError("need at least one scored pair")

    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    labels = set(a) | set(b)
    pe = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)

    if pe == 1.0:
        # Every score is the same single label for both raters: po == 1.0
        # too in that case, so agreement is perfect and (po - pe) / (1 -
        # pe) is the undefined 0/0 -- defined as 1.0 rather than raising,
        # since "always agree" is exactly what happened.
        return 1.0
    return (po - pe) / (1 - pe)


def write_human_review_worksheet(
    questions: list[EvalQuestion], scored: list[ScoredQuestionRun], path: Path
) -> Path:
    """A CSV worksheet for the 15-question sample: question, the
    system's answer, the gold facts/sources, the judge's own score --
    and a blank `human_score` column for you to fill in by hand before
    computing agreement against the judge column. `scored` is looked up
    by `question_id`; a sampled question with no matching scored run
    (report predates the question, or the run errored) gets blank
    answer/judge columns rather than being dropped, so the worksheet
    still has all 15 rows to fill in manually if needed."""
    sample = select_human_review_sample(questions)
    scored_by_id = {s.question_id: s for s in scored}

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_WORKSHEET_HEADER)
        for question in sample:
            run = scored_by_id.get(question.id)
            judge_score: int | str = ""
            judge_rationale = ""
            if run is not None and isinstance(run.layer2, JudgeResult):
                judge_score = run.layer2.score
                judge_rationale = run.layer2.rationale
            writer.writerow(
                [
                    question.id,
                    question.category,
                    question.question,
                    run.answer if run is not None else "",
                    "; ".join(question.expected_answer_points),
                    "; ".join(
                        f"{g.document} {g.section or ''}".strip() for g in question.gold_sources
                    ),
                    judge_score,
                    judge_rationale,
                    "",
                ]
            )
    return path
