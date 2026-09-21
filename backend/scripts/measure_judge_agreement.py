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
import io
import sys
from pathlib import Path

from app.eval.agreement import cohens_kappa, percent_agreement


def _int_column(rows: list[dict[str, str]], column: str) -> list[int]:
    blank = [r["question_id"] for r in rows if not r[column].strip()]
    if blank:
        raise SystemExit(f"{column} is empty for: {', '.join(blank)} -- fill in every row first")
    return [int(r[column]) for r in rows]


def _run(path: Path) -> None:
    # utf-8-sig transparently strips a BOM if present and is otherwise
    # identical to utf-8; sniff the delimiter rather than assume `;`
    # (write_human_review_worksheet's choice, Excel-friendly for a
    # German/Italian/etc. locale) since Excel's Save/Save As can rewrite
    # a CSV with a different one depending on the system's regional
    # settings. Sniff off just the header line, then hand the WHOLE text
    # to csv.DictReader via StringIO (not text.splitlines()) so a
    # multi-line quoted answer field is parsed correctly instead of
    # being torn apart at its embedded newlines.
    text = path.read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=";,")
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))

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
