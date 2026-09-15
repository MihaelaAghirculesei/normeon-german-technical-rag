"""Compute human-vs-judge agreement from a filled-in human_review.csv
(plan, Giorno 18's "step almost nobody does"): score the 15 rows'
`human_score` column by hand yourself, then run this to get percent
agreement and Cohen's kappa against the judge's own `judge_score`
column -- and paste the result into docs/EVALUATION.md yourself.

    cd backend
    .venv/Scripts/python scripts/measure_judge_agreement.py \\
        eval/reports/<hash>.human_review.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from app.eval.agreement import cohens_kappa, percent_agreement


def _int_column(rows: list[dict[str, str]], column: str) -> list[int]:
    blank = [r["question_id"] for r in rows if not r[column].strip()]
    if blank:
        raise SystemExit(f"{column} is empty for: {', '.join(blank)} -- fill in every row first")
    return [int(r[column]) for r in rows]


def _run(path: Path) -> None:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    human = _int_column(rows, "human_score")
    judge = _int_column(rows, "judge_score")

    print(f"n = {len(rows)}")
    print(f"percent agreement = {percent_agreement(human, judge):.1%}")
    print(f"Cohen's kappa      = {cohens_kappa(human, judge):.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "worksheet", help="a filled-in *.human_review.csv from score_eval_report.py"
    )
    args = parser.parse_args()
    _run(Path(args.worksheet))
    return 0


if __name__ == "__main__":
    sys.exit(main())
