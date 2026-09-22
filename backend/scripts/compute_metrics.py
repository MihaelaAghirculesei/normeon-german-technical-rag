"""Compute the Giorno 19 retrieval/citation/outcome metrics for an
already-scored eval report and write `<hash>.metrics.json` next to it.

    cd backend
    .venv/Scripts/python scripts/compute_metrics.py eval/reports/<hash>.json

Needs a report already scored by scripts/score_eval_report.py (reads
its `<hash>.scored.json` from the same directory). Pure -- no DB, no
LLM, no network; safe to re-run for free.
"""

import argparse
import sys
from pathlib import Path

from app.eval.dataset import DATASET_DIR, load_questions
from app.eval.metrics import compute_metrics
from app.eval.models import EvalReport
from app.eval.scoring import ScoredReport


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="path to an eval/reports/<hash>.json file")
    parser.add_argument(
        "--questions",
        default=str(DATASET_DIR / "questions.yaml"),
        help="the question set the report was run against (default: the real 50-question set)",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    report = EvalReport.model_validate_json(report_path.read_text(encoding="utf-8"))

    scored_path = report_path.parent / f"{report.config_hash}.scored.json"
    if not scored_path.exists():
        print(
            f"error: no scored report at {scored_path} -- run "
            "scripts/score_eval_report.py first",
            file=sys.stderr,
        )
        return 1
    scored = ScoredReport.model_validate_json(scored_path.read_text(encoding="utf-8"))
    questions = load_questions(Path(args.questions))

    metrics = compute_metrics(report, scored, questions)

    metrics_path = report_path.parent / f"{report.config_hash}.metrics.json"
    metrics_path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")

    print(metrics.model_dump_json(indent=2))
    print(f"metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
