"""Unit tests for app/eval/matrix (plan, Giorno 20): the markdown/HTML
comparison table across the experiment matrix's configurations."""

from __future__ import annotations

import pytest

from app.eval.matrix import MatrixRow, build_html_report, build_markdown_report
from app.eval.metrics import EvalMetrics
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
