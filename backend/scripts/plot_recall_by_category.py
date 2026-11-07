"""Draw the README's results chart from the ablation matrix's reports.

Every row is found by its configuration, not by a hash typed in here: the
baseline, plus the one knob each experiment changes. A configuration with
failed questions is left out (E8 ran out of quota after 13 answers).

Run with:
    cd backend
    .venv/Scripts/python scripts/plot_recall_by_category.py \\
        --reports-dir eval/reports/matrix --out-dir ../docs/img
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.eval.chart import ChartRow, build_svg, recall_cells
from app.eval.metrics import QuestionOutcome

BASELINE: dict[str, Any] = {
    "chunking_strategy": "structural",
    "retrieval_mode": "hybrid",
    "reranker": "cross_encoder",
    "top_k": 5,
    "fts_match": "all",
    "embedding_provider": "e5_local",
    "llm_model": "gemini-flash-lite-latest",
}
# (label, the knobs that differ from the baseline, bold in the chart)
ROWS: list[tuple[str, dict[str, Any], bool]] = [
    ("Baseline (e5, reranker)", {}, True),
    ("E1 fixed 500-token chunks", {"chunking_strategy": "fixed_500"}, False),
    ("E2 vector only", {"retrieval_mode": "vector"}, False),
    ("E3 full-text only", {"retrieval_mode": "fts"}, False),
    ("E4 no reranker", {"reranker": "noop"}, False),
    ("E5 top-k 3", {"top_k": 3}, False),
    ("E6 top-k 10", {"top_k": 10}, False),
    ("E7 Gemini embeddings", {"embedding_provider": "gemini"}, False),
    ("E9 full-text, any term", {"fts_match": "any"}, False),
    ("Deployed (Gemini, no reranker)", {"embedding_provider": "gemini", "reranker": "noop"}, True),
]
FOOTNOTE = (
    "Ablation matrix, 40 answerable questions (docs/EVALUATION.md). "
    "E8 is left out: its run is incomplete."
)


def _config(report_path: Path) -> dict[str, Any]:
    config: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))["config"]
    if config.get("fts_match") is None:  # reports from before the knob
        config["fts_match"] = "all"
    return config


def collect(reports_dir: Path) -> list[ChartRow]:
    found: dict[str, ChartRow] = {}
    for metrics_path in sorted(reports_dir.glob("*.metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = _config(metrics_path.with_name(metrics_path.name.replace(".metrics", "")))
        for label, knobs, emphasis in ROWS:
            wanted = {**BASELINE, **knobs}
            if all(config.get(key) == value for key, value in wanted.items()):
                if metrics["n_errors"]:
                    print(f"skipped {label}: {metrics['n_errors']} failed questions")
                    break
                if label in found:
                    raise SystemExit(f"two reports match {label!r}")
                outcomes = [QuestionOutcome.model_validate(o) for o in metrics["outcomes"]]
                found[label] = ChartRow(label, recall_cells(outcomes), emphasis)
    missing = [label for label, _, _ in ROWS if label not in found]
    if missing:
        raise SystemExit(f"no complete report for: {', '.join(missing)}")
    return [found[label] for label, _, _ in ROWS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports-dir", type=Path, default=Path("eval/reports/matrix"))
    parser.add_argument("--out-dir", type=Path, default=Path("../docs/img"))
    args = parser.parse_args()

    rows = collect(args.reports_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        path = args.out_dir / f"recall-by-category-{theme}.svg"
        path.write_text(build_svg(rows, theme, footnote=FOOTNOTE), encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    for row in rows:
        print(
            f"  {row.label:32} "
            + "  ".join(f"{k}={c.hits}/{c.total}" for k, c in row.cells.items())
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
