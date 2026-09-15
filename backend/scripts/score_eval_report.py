"""Score an existing eval report's answers with the two-layer
correctness metric (plan, Giorno 18) and export a 15-question worksheet
for human review.

    cd backend
    DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \\
        .venv/Scripts/python scripts/score_eval_report.py eval/reports/<hash>.json

Writes, next to the report:
  - <hash>.scored.json      (Layer 1 + Layer 2 per question)
  - <hash>.human_review.csv (15-question worksheet, a blank human_score
    column -- fill it in by hand, then run
    scripts/measure_judge_agreement.py on it)

The judge is whichever LlmClient LLM_PROVIDER/LLM_API_BASE_URL/LLM_MODEL
resolve to -- there is no separate judge-model setting; point them at
the model you want as judge before running this. Needs a real (non-
"fake") LLM provider configured to be meaningful.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.api.deps import get_llm_client
from app.eval.agreement import write_human_review_worksheet
from app.eval.dataset import DATASET_DIR, load_questions
from app.eval.models import EvalReport
from app.eval.scoring import JudgeResult, score_report, write_scored_report


async def _run(report_path: Path, questions_path: Path) -> None:
    report = EvalReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    questions = load_questions(questions_path)

    scored = await score_report(report, questions, get_llm_client())
    scored_path = write_scored_report(scored, reports_dir=report_path.parent)

    worksheet_path = report_path.parent / f"{report.config_hash}.human_review.csv"
    write_human_review_worksheet(questions, scored.results, worksheet_path)

    layer1_pass = sum(1 for r in scored.results if r.layer1.all_present)
    layer2_scores = [r.layer2.score for r in scored.results if isinstance(r.layer2, JudgeResult)]
    judge_errors = len(scored.results) - len(layer2_scores)

    print(f"{len(scored.results)} answers scored")
    print(f"Layer 1 all-points-present: {layer1_pass}/{len(scored.results)}")
    if layer2_scores:
        print(f"Layer 2 mean score: {sum(layer2_scores) / len(layer2_scores):.2f}")
    if judge_errors:
        print(f"Layer 2 judge errors (malformed response): {judge_errors}")
    print(f"scored report          -> {scored_path}")
    print(f"human review worksheet -> {worksheet_path}")


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
    asyncio.run(_run(Path(args.report), Path(args.questions)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
