"""Unit tests for app/eval/matrix (plan, Giorno 20): the markdown/HTML
comparison table across the experiment matrix's configurations."""

from __future__ import annotations

import pytest

from app.eval.matrix import MatrixRow, build_html_report, build_markdown_report
from app.eval.metrics import EvalMetrics, QuestionOutcome
from app.eval.models import EvalConfig


def _config(chunking: str, mode: str, top_k: int = 5) -> EvalConfig:
    return EvalConfig(
        chunking_strategy=chunking,  # type: ignore[arg-type]
        retrieval_mode=mode,  # type: ignore[arg-type]
        reranker="cross_encoder",
        top_k=top_k,
        embedding_provider="e5_local",
        llm_provider="fake",
        llm_model="fake",
    )


def _metrics(**overrides: object) -> EvalMetrics:
    defaults: dict[str, object] = {
        "config_hash": "h",
        "n_questions": 50,
        "n_errors": 0,
        "recall_at_k": 0.8,
        "mrr": 0.7,
        "precision_at_k": 0.3,
        "citation_precision": 0.6,
        "hallucinated_citation_rate": 0.0,
        "answer_accuracy": 0.7,
        "correct_abstention_rate": 0.9,
        "false_abstention_rate": 0.1,
        "latency_ms_p50": {"retrieval_ms": 100.0, "generation_ms": 500.0, "total_ms": 600.0},
        "latency_ms_p95": {"retrieval_ms": 200.0, "generation_ms": 900.0, "total_ms": 1100.0},
        "cost_per_query_avg": 0.001,
    }
    defaults.update(overrides)
    return EvalMetrics.model_validate(defaults)


def _row(label: str, **metric_overrides: object) -> MatrixRow:
    return MatrixRow(
        label=label,
        config=_config("structural", "hybrid"),
        metrics=_metrics(**metric_overrides),
    )


def test_build_markdown_report_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        build_markdown_report([])


def test_build_html_report_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        build_html_report([])


def test_markdown_report_has_one_row_per_config() -> None:
    rows = [_row("A"), _row("B")]
    report = build_markdown_report(rows)
    assert "| A |" in report
    assert "| B |" in report
    assert report.count("\n| ") >= 3  # header + separator + 2 data rows


def test_markdown_report_bolds_the_higher_is_better_winner() -> None:
    rows = [_row("low", recall_at_k=0.5), _row("high", recall_at_k=0.9)]
    report = build_markdown_report(rows)
    assert "**0.900**" in report
    assert "**0.500**" not in report


def test_markdown_report_bolds_the_lower_is_better_winner() -> None:
    slow_p50 = {"retrieval_ms": 1.0, "generation_ms": 1.0, "total_ms": 2000.0}
    fast_p50 = {"retrieval_ms": 1.0, "generation_ms": 1.0, "total_ms": 200.0}
    rows = [_row("slow", latency_ms_p50=slow_p50), _row("fast", latency_ms_p50=fast_p50)]
    report = build_markdown_report(rows)
    assert "**200.000**" in report
    assert "**2000.000**" not in report


def test_markdown_report_none_metric_renders_as_dash_and_never_wins() -> None:
    rows = [_row("has_cost", cost_per_query_avg=0.002), _row("no_cost", cost_per_query_avg=None)]
    report = build_markdown_report(rows)
    assert "–" in report
    assert "**0.002**" in report


def test_html_report_contains_all_labels_and_is_valid_enough() -> None:
    rows = [_row("A"), _row("B")]
    report = build_html_report(rows)
    assert "<table>" in report
    assert "<td>A</td>" in report
    assert "<td>B</td>" in report


def test_html_report_marks_the_winner_cell() -> None:
    rows = [_row("low", recall_at_k=0.5), _row("high", recall_at_k=0.9)]
    report = build_html_report(rows)
    assert '<td class="winner"><strong>0.900</strong></td>' in report


# --- category breakdown, intervals, paired comparison -------------------------


def _outcomes(
    correct: dict[str, bool], category: str = "requirement_lookup", gold: bool | None = True
) -> list[QuestionOutcome]:
    return [
        QuestionOutcome(question_id=qid, category=category, answer_correct=ok, gold_hit=gold)  # type: ignore[arg-type]
        for qid, ok in correct.items()
    ]


def _ids(n: int) -> list[str]:
    return [f"Q{i:03d}" for i in range(1, n + 1)]


def test_overall_table_shows_intervals_and_error_counts() -> None:
    row = _row("A", answer_accuracy_ci95=(0.6, 0.84), n_errors=2, n_judge_errors=3)
    report = build_markdown_report([row])
    assert "[0.600, 0.840]" in report
    assert "| 2 | 3 |" in report


def test_accuracy_by_category_counts_hits_per_category() -> None:
    outcomes = _outcomes({"Q001": True, "Q002": False}) + _outcomes(
        {"Q011": True, "Q012": True}, category="code_lookup"
    )
    report = build_markdown_report([_row("A", outcomes=outcomes)])
    assert "## Answer accuracy by category" in report
    assert "| A | **0.500 (1/2)** | **1.000 (2/2)** |" in report


def test_recall_by_category_skips_categories_without_gold() -> None:
    outcomes = _outcomes({"Q001": True}) + _outcomes(
        {"Q031": True}, category="unanswerable", gold=None
    )
    report = build_markdown_report([_row("A", outcomes=outcomes)])
    recall_section = report.split("## Recall@k by category")[1].split("##")[0]
    assert "requirement_lookup" in recall_section
    assert "unanswerable" not in recall_section


def test_paired_table_flags_a_supported_improvement() -> None:
    ids = _ids(50)
    base = _outcomes({q: i < 30 for i, q in enumerate(ids)})
    better = _outcomes({q: i < 42 for i, q in enumerate(ids)})
    report = build_markdown_report(
        [_row("base", outcomes=base), _row("better", outcomes=better)], baseline_label="base"
    )
    assert "## Paired difference vs. baseline (base)" in report
    assert "**+0.240 ***" in report


def test_paired_table_does_not_flag_noise() -> None:
    ids = _ids(50)
    base = _outcomes({q: i < 30 for i, q in enumerate(ids)})
    noisy = _outcomes({q: (i < 29 or i == 45) for i, q in enumerate(ids)})
    report = build_markdown_report([_row("base", outcomes=base), _row("noisy", outcomes=noisy)])
    assert "| noisy | +0.000 |" in report


def test_paired_table_is_omitted_without_outcomes() -> None:
    report = build_markdown_report([_row("A"), _row("B")])
    assert "Paired difference" not in report


def test_unknown_baseline_is_an_error() -> None:
    with pytest.raises(ValueError, match="baseline"):
        build_markdown_report([_row("A")], baseline_label="nope")


def test_html_report_renders_every_section() -> None:
    ids = _ids(10)
    rows = [
        _row("base", outcomes=_outcomes({q: True for q in ids})),
        _row("other", outcomes=_outcomes({q: False for q in ids})),
    ]
    report = build_html_report(rows)
    for title in ("Overall", "Answer accuracy by category", "Recall@k by category"):
        assert f"<h2>{title}</h2>" in report
    assert "Paired difference vs. baseline (base)" in report


def test_notes_are_rendered_in_both_formats() -> None:
    note = "Latency columns include rerank-cache hits"
    assert f"> {note}" in build_markdown_report([_row("A")], notes=[note])
    assert f'<p class="note">{note}</p>' in build_html_report([_row("A")], notes=[note])
