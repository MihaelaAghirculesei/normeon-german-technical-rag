"""The Giorno 20 comparison report: one row per configuration in the
experiment matrix, rendered as markdown and as HTML from the same table
model, so a reader (or the ADR that follows) can say concretely which
configuration wins, on which category, and whether the margin is larger
than a 50-question sample can resolve.

Four tables:

1. Overall -- every metric, the per-column winner bolded, plus 95%
   bootstrap intervals and the error counts that qualify the numbers.
2. Answer accuracy by category.
3. Recall@k by category (categories with gold sources only).
4. Paired difference vs. a baseline row -- question-by-question
   bootstrap, since every configuration answers the same questions.

Pure -- no I/O beyond `write_matrix_report`, which dumps two strings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from app.eval.metrics import EvalMetrics, QuestionOutcome
from app.eval.models import CATEGORIES, EvalConfig
from app.eval.runner import REPORTS_DIR
from app.eval.stats import paired_bootstrap_diff

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


@dataclass(frozen=True, slots=True)
class _Cell:
    text: str
    winner: bool = False


@dataclass(frozen=True, slots=True)
class _Table:
    title: str
    header: list[str]
    rows: list[list[_Cell]]
    note: str | None = None
    # leading label columns, left-aligned in HTML; the rest are numbers
    label_columns: int = 1


def _value(row: MatrixRow, column: str) -> float | None:
    if column == "latency_p50_total_ms":
        return row.metrics.latency_ms_p50.get("total_ms")
    if column == "latency_p95_total_ms":
        return row.metrics.latency_ms_p95.get("total_ms")
    return getattr(row.metrics, column)  # type: ignore[no-any-return]


def _best(values: list[float | None], *, higher_is_better: bool) -> float | None:
    """The winning value among `values`; `None`s (a metric that couldn't
    be computed for that run) never win."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return max(present) if higher_is_better else min(present)


def _format(value: float | None) -> str:
    if value is None:
        return "–"
    return f"{value:.3f}"


def _format_ci(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "–"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _scored_cell(value: float | None, best: float | None, text: str | None = None) -> _Cell:
    return _Cell(
        text if text is not None else _format(value),
        winner=value is not None and value == best,
    )


def _overall_table(rows: list[MatrixRow]) -> _Table:
    best = {
        column: _best(
            [_value(r, column) for r in rows], higher_is_better=column in _HIGHER_IS_BETTER
        )
        for column in _METRIC_COLUMNS
    }
    header = (
        ["Config", "Chunking", "Retrieval", "top_k"]
        + list(_METRIC_COLUMNS)
        + ["answer_accuracy 95% CI", "recall_at_k 95% CI", "errors", "judge errors"]
    )
    body = []
    for row in rows:
        cells = [
            _Cell(row.label),
            _Cell(row.config.chunking_strategy),
            _Cell(row.config.retrieval_mode),
            _Cell(str(row.config.top_k)),
        ]
        cells += [_scored_cell(_value(row, c), best[c]) for c in _METRIC_COLUMNS]
        cells += [
            _Cell(_format_ci(row.metrics.answer_accuracy_ci95)),
            _Cell(_format_ci(row.metrics.recall_at_k_ci95)),
            _Cell(str(row.metrics.n_errors)),
            _Cell(str(row.metrics.n_judge_errors)),
        ]
        body.append(cells)
    return _Table(
        title="Overall",
        header=header,
        rows=body,
        label_columns=4,
        note=(
            "Bold marks the best value per column. Intervals are 95% percentile "
            "bootstrap over questions; overlapping intervals alone do not decide a "
            "winner -- see the paired table below."
        ),
    )


def _rate(
    outcomes: list[QuestionOutcome], category: str, pick: Callable[[QuestionOutcome], bool | None]
) -> tuple[float, int, int] | None:
    values = [v for o in outcomes if o.category == category and (v := pick(o)) is not None]
    if not values:
        return None
    hits = sum(values)
    return hits / len(values), hits, len(values)


def _by_category_table(
    rows: list[MatrixRow], title: str, pick: Callable[[QuestionOutcome], bool | None]
) -> _Table | None:
    rates = {
        (row.label, category): _rate(row.metrics.outcomes, category, pick)
        for row in rows
        for category in CATEGORIES
    }
    categories = [c for c in CATEGORIES if any(rates[(r.label, c)] for r in rows)]
    if not categories:
        return None
    best = {
        c: _best(
            [rate[0] if (rate := rates[(r.label, c)]) else None for r in rows],
            higher_is_better=True,
        )
        for c in categories
    }
    body = []
    for row in rows:
        cells = [_Cell(row.label)]
        for category in categories:
            rate = rates[(row.label, category)]
            if rate is None:
                cells.append(_Cell("–"))
            else:
                value, hits, n = rate
                cells.append(_scored_cell(value, best[category], f"{value:.3f} ({hits}/{n})"))
        body.append(cells)
    return _Table(title=title, header=["Config", *categories], rows=body)


def _paired(
    baseline: list[QuestionOutcome],
    candidate: list[QuestionOutcome],
    pick: Callable[[QuestionOutcome], bool | None],
) -> tuple[float, float, float] | None:
    base = {o.question_id: pick(o) for o in baseline}
    pairs = [
        (float(b), float(c))
        for o in candidate
        if (c := pick(o)) is not None and (b := base.get(o.question_id)) is not None
    ]
    return paired_bootstrap_diff([b for b, _ in pairs], [c for _, c in pairs])


def _format_diff(result: tuple[float, float, float] | None) -> tuple[_Cell, _Cell]:
    if result is None:
        return _Cell("–"), _Cell("–")
    diff, lo, hi = result
    significant = lo > 0 or hi < 0
    return _Cell(f"{diff:+.3f}{' *' if significant else ''}", winner=significant), _Cell(
        f"[{lo:+.3f}, {hi:+.3f}]"
    )


def _paired_table(rows: list[MatrixRow], baseline: MatrixRow) -> _Table | None:
    if not baseline.metrics.outcomes:
        return None
    body = []
    for row in rows:
        if row is baseline or not row.metrics.outcomes:
            continue
        acc = _format_diff(
            _paired(baseline.metrics.outcomes, row.metrics.outcomes, lambda o: o.answer_correct)
        )
        rec = _format_diff(
            _paired(baseline.metrics.outcomes, row.metrics.outcomes, lambda o: o.gold_hit)
        )
        body.append([_Cell(row.label), *acc, *rec])
    if not body:
        return None
    return _Table(
        title=f"Paired difference vs. baseline ({baseline.label})",
        header=["Config", "Δ answer_accuracy", "95% CI", "Δ recall_at_k", "95% CI"],
        rows=body,
        note=(
            "Each configuration minus the baseline, over the questions both answered, "
            "with a paired bootstrap over questions. `*` (bold): the 95% interval "
            "excludes zero -- a difference this sample supports; anything else is "
            "within noise."
        ),
    )


def _tables(rows: list[MatrixRow], baseline_label: str | None) -> list[_Table]:
    if not rows:
        raise ValueError("need at least one row")
    baseline = rows[0]
    if baseline_label is not None:
        matches = [r for r in rows if r.label == baseline_label]
        if not matches:
            raise ValueError(f"baseline {baseline_label!r} is not one of the rows")
        baseline = matches[0]
    candidates = [
        _overall_table(rows),
        _by_category_table(rows, "Answer accuracy by category", lambda o: o.answer_correct),
        _by_category_table(rows, "Recall@k by category", lambda o: o.gold_hit),
        _paired_table(rows, baseline),
    ]
    return [t for t in candidates if t is not None]


# --- markdown -----------------------------------------------------------------


def _md_table(table: _Table) -> list[str]:
    lines = [f"## {table.title}", ""]
    if table.note:
        lines += [table.note, ""]
    lines.append("| " + " | ".join(table.header) + " |")
    lines.append("| " + " | ".join(["---"] * len(table.header)) + " |")
    for row in table.rows:
        lines.append(
            "| " + " | ".join(f"**{c.text}**" if c.winner else c.text for c in row) + " |"
        )
    lines.append("")
    return lines


def build_markdown_report(
    rows: list[MatrixRow], baseline_label: str | None = None, notes: list[str] | None = None
) -> str:
    lines = ["# Evaluation matrix", ""]
    for note in notes or []:
        lines += [f"> {note}", ""]
    for table in _tables(rows, baseline_label):
        lines += _md_table(table)
    return "\n".join(lines)


# --- HTML ---------------------------------------------------------------------


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html_table(table: _Table) -> str:
    def th(i: int, text: str) -> str:
        cls = ' class="label"' if i < table.label_columns else ""
        return f"<th{cls}>{_html_escape(text)}</th>"

    def td(i: int, cell: _Cell) -> str:
        text = _html_escape(cell.text)
        if cell.winner:
            return f'<td class="winner"><strong>{text}</strong></td>'
        if i < table.label_columns:
            return f"<td>{text}</td>"
        return f'<td class="num">{text}</td>'

    thead = "".join(th(i, h) for i, h in enumerate(table.header))
    body = "\n".join(
        "<tr>" + "".join(td(i, c) for i, c in enumerate(row)) + "</tr>" for row in table.rows
    )
    note = f"<p>{_html_escape(table.note)}</p>\n" if table.note else ""
    return (
        f"<h2>{_html_escape(table.title)}</h2>\n{note}"
        f"<table>\n<thead><tr>{thead}</tr></thead>\n<tbody>\n{body}\n</tbody>\n</table>\n"
    )


def build_html_report(
    rows: list[MatrixRow], baseline_label: str | None = None, notes: list[str] | None = None
) -> str:
    sections = "\n".join(_html_table(t) for t in _tables(rows, baseline_label))
    notes_html = "".join(f'<p class="note">{_html_escape(note)}</p>\n' for note in notes or [])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Evaluation matrix</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  table {{ border-collapse: collapse; margin-bottom: 2rem; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: right; }}
  th.label, td:not(.num):not(.winner) {{ text-align: left; }}
  th {{ background: #f0f0f0; }}
  td.winner {{ background: #e6f4ea; }}
  p {{ max-width: 60rem; color: #444; }}
  p.note {{ border-left: 3px solid #c90; padding-left: 0.6rem; }}
</style>
</head>
<body>
<h1>Evaluation matrix</h1>
{notes_html}{sections}
</body>
</html>
"""


def write_matrix_report(
    rows: list[MatrixRow],
    reports_dir: Path = REPORTS_DIR,
    *,
    baseline_label: str | None = None,
    notes: list[str] | None = None,
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / "matrix.md"
    html_path = reports_dir / "matrix.html"
    md_path.write_text(build_markdown_report(rows, baseline_label, notes), encoding="utf-8")
    html_path.write_text(build_html_report(rows, baseline_label, notes), encoding="utf-8")
    return md_path, html_path
