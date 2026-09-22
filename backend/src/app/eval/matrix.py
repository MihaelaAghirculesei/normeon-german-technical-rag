"""The Giorno 20 comparison report: one row per configuration in the
experiment matrix, one markdown table and one HTML table, both showing
the same numbers so a reader (or the ADR that follows) can say
concretely which configuration wins and on what.

Pure -- no I/O beyond the two `write_*` helpers, which just dump a
string to a path; building the tables themselves never touches a
report, the DB or an LLM.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from app.eval.metrics import EvalMetrics
from app.eval.models import EvalConfig
from app.eval.runner import REPORTS_DIR

# Which direction is "better" for each metric column, so the winning
# cell in each column can be marked -- not just printed, since a bare
# number doesn't say "high is good here but low is good over there"
# (recall vs. latency, for instance).
_HIGHER_IS_BETTER = (
    "recall_at_k",
    "mrr",
    "precision_at_k",
    "citation_precision",
    "answer_accuracy",
    "correct_abstention_rate",
)
_LOWER_IS_BETTER = (
    "hallucinated_citation_rate",
    "false_abstention_rate",
    "latency_p50_total_ms",
    "latency_p95_total_ms",
    "cost_per_query_avg",
)
_METRIC_COLUMNS = _HIGHER_IS_BETTER + _LOWER_IS_BETTER


class MatrixRow(BaseModel):
    label: str
    config: EvalConfig
    metrics: EvalMetrics


def _value(row: MatrixRow, column: str) -> float | None:
    if column == "latency_p50_total_ms":
        return row.metrics.latency_ms_p50.get("total_ms")
    if column == "latency_p95_total_ms":
        return row.metrics.latency_ms_p95.get("total_ms")
    return getattr(row.metrics, column)  # type: ignore[no-any-return]


def _winners(rows: list[MatrixRow]) -> dict[str, float]:
    """The best value present for each metric column, across rows --
    `None`s (a metric that couldn't be computed for that run) never
    win. Empty if no row has a value for that column."""
    winners: dict[str, float] = {}
    for column in _METRIC_COLUMNS:
        values = [v for row in rows if (v := _value(row, column)) is not None]
        if not values:
            continue
        winners[column] = max(values) if column in _HIGHER_IS_BETTER else min(values)
    return winners


def _format(value: float | None) -> str:
    if value is None:
        return "–"
    return f"{value:.3f}"


def build_markdown_report(rows: list[MatrixRow]) -> str:
    if not rows:
        raise ValueError("need at least one row")
    winners = _winners(rows)

    header = ["Config", "Chunking", "Retrieval", "top_k"] + list(_METRIC_COLUMNS)
    lines = [
        "# Evaluation matrix",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        cells = [
            row.label,
            row.config.chunking_strategy,
            row.config.retrieval_mode,
            str(row.config.top_k),
        ]
        for column in _METRIC_COLUMNS:
            value = _value(row, column)
            text = _format(value)
            if value is not None and value == winners.get(column):
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_html_report(rows: list[MatrixRow]) -> str:
    if not rows:
        raise ValueError("need at least one row")
    winners = _winners(rows)

    header = ["Config", "Chunking", "Retrieval", "top_k"] + list(_METRIC_COLUMNS)
    thead = "".join(f"<th>{_html_escape(h)}</th>" for h in header)

    body_rows = []
    for row in rows:
        cells = [
            row.label,
            row.config.chunking_strategy,
            row.config.retrieval_mode,
            str(row.config.top_k),
        ]
        tds = "".join(f"<td>{_html_escape(c)}</td>" for c in cells)
        for column in _METRIC_COLUMNS:
            value = _value(row, column)
            text = _format(value)
            if value is not None and value == winners.get(column):
                tds += f'<td class="winner"><strong>{text}</strong></td>'
            else:
                tds += f"<td>{text}</td>"
        body_rows.append(f"<tr>{tds}</tr>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Evaluation matrix</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: right; }}
  th:nth-child(-n+4), td:nth-child(-n+4) {{ text-align: left; }}
  th {{ background: #f0f0f0; }}
  td.winner {{ background: #e6f4ea; }}
</style>
</head>
<body>
<h1>Evaluation matrix</h1>
<table>
<thead><tr>{thead}</tr></thead>
<tbody>
{"".join(body_rows)}
</tbody>
</table>
</body>
</html>
"""


def write_matrix_report(
    rows: list[MatrixRow], reports_dir: Path = REPORTS_DIR
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / "matrix.md"
    html_path = reports_dir / "matrix.html"
    md_path.write_text(build_markdown_report(rows), encoding="utf-8")
    html_path.write_text(build_html_report(rows), encoding="utf-8")
    return md_path, html_path
