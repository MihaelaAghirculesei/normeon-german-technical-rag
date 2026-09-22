"""Unit tests for app/eval/metrics (plan, Giorno 19): retrieval,
citation and outcome metrics computed over a real eval run. Pure --
every fixture here is a synthetic EvalQuestion/QuestionRun/
ScoredQuestionRun, no report/DB/LLM involved."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from app.eval.metrics import (
    EvalMetrics,
    answer_accuracy,
    citation_precision,
    compute_metrics,
    correct_abstention_rate,
    cost_per_query_avg,
    false_abstention_rate,
    hallucinated_citation_rate,
    latency_percentiles,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from app.eval.models import (
    CitationRecord,
    EvalConfig,
    EvalQuestion,
    EvalReport,
    GoldSource,
    QuestionRun,
    RetrievedRecord,
)
from app.eval.scoring import JudgeError, JudgeResult, Layer1Result, ScoredQuestionRun, ScoredReport


def _question(**overrides: object) -> EvalQuestion:
    defaults: dict[str, object] = {
        "id": "Q001",
        "category": "requirement_lookup",
        "question": "Frage?",
        "gold_sources": [GoldSource(document="StVZO.pdf", section="§41", page=40)],
        "should_abstain": False,
    }
    defaults.update(overrides)
    return EvalQuestion.model_validate(defaults)


def _retrieved(
    filename: str = "StVZO.pdf", page: int = 40, section: str | None = "§41"
) -> RetrievedRecord:
    return RetrievedRecord(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=filename,
        page_from=page,
        page_to=page,
        section_path=section,
    )


def _citation(
    filename: str = "StVZO.pdf", page: int = 40, section: str | None = "§41"
) -> CitationRecord:
    return CitationRecord(
        marker="S1",
        document_id=uuid.uuid4(),
        filename=filename,
        page_from=page,
        page_to=page,
        section_path=section,
    )


def _run(
    question_id: str = "Q001",
    *,
    retrieved: list[RetrievedRecord] | None = None,
    citations: list[CitationRecord] | None = None,
    invented_citations: int = 0,
    abstained: bool = False,
    error: str | None = None,
    retrieval_ms: float = 10.0,
    generation_ms: float = 5.0,
    total_ms: float = 15.0,
    cost_usd: float | None = 0.001,
) -> QuestionRun:
    return QuestionRun(
        question_id=question_id,
        category="requirement_lookup",
        answer="Antwort" if error is None else "",
        abstained=abstained,
        citations=citations or [],
        invented_citations=invented_citations,
        retrieved=retrieved or [],
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        total_ms=total_ms,
        cost_usd=cost_usd,
        prompt_name="answer_de.v1",
        prompt_sha256="deadbeef",
        model="fake",
        error=error,
    )


def _scored(
    question_id: str = "Q001",
    *,
    points_missing: list[str] | None = None,
    layer2: JudgeResult | JudgeError | None = None,
) -> ScoredQuestionRun:
    return ScoredQuestionRun(
        question_id=question_id,
        category="requirement_lookup",
        answer="Antwort",
        layer1=Layer1Result(points_found=[], points_missing=points_missing or []),
        layer2=layer2 or JudgeResult(score=2, rationale="ok"),
    )


# --- recall_at_k / mean_reciprocal_rank / precision_at_k -------------------


def test_recall_at_k_hit_and_miss() -> None:
    hit = _question(id="Q001")
    miss = _question(id="Q002")
    results = [
        _run("Q001", retrieved=[_retrieved(page=40)]),
        _run("Q002", retrieved=[_retrieved(page=99, section="§99")]),
    ]
    assert recall_at_k([hit, miss], results) == 0.5


def test_recall_at_k_skips_gold_free_and_errored_questions() -> None:
    unanswerable = _question(id="Q001", gold_sources=[], should_abstain=True)
    errored = _question(id="Q002")
    results = [
        _run("Q001", abstained=True),
        _run("Q002", error="boom"),
    ]
    assert recall_at_k([unanswerable, errored], results) is None


def test_recall_at_k_page_within_tolerance() -> None:
    q = _question(id="Q001", gold_sources=[GoldSource(document="StVZO.pdf", page=40)])
    results = [_run("Q001", retrieved=[_retrieved(page=41, section=None)])]
    assert recall_at_k([q], results) == 1.0


def test_recall_at_k_page_outside_tolerance_no_section() -> None:
    q = _question(id="Q001", gold_sources=[GoldSource(document="StVZO.pdf", page=40)])
    results = [_run("Q001", retrieved=[_retrieved(page=43, section=None)])]
    assert recall_at_k([q], results) == 0.0


def test_recall_at_k_document_mismatch_never_matches() -> None:
    q = _question(id="Q001", gold_sources=[GoldSource(document="StVZO.pdf", page=40)])
    results = [_run("Q001", retrieved=[_retrieved(filename="FZV.pdf", page=40)])]
    assert recall_at_k([q], results) == 0.0


def test_recall_at_k_gold_with_no_section_or_page_matches_on_document() -> None:
    q = _question(id="Q001", gold_sources=[GoldSource(document="Lastenheft-EPS-v1.2.docx")])
    results = [
        _run("Q001", retrieved=[_retrieved(filename="Lastenheft-EPS-v1.2.docx", section=None)])
    ]
    assert recall_at_k([q], results) == 1.0


def test_mean_reciprocal_rank_first_and_third_position() -> None:
    q1 = _question(id="Q001")
    q2 = _question(id="Q002")
    results = [
        _run("Q001", retrieved=[_retrieved(page=40)]),
        _run(
            "Q002",
            retrieved=[
                _retrieved(page=1, section="§1"),
                _retrieved(page=2, section="§2"),
                _retrieved(page=40),
            ],
        ),
    ]
    assert mean_reciprocal_rank([q1, q2], results) == pytest.approx((1.0 + 1 / 3) / 2)


def test_mean_reciprocal_rank_zero_when_no_match() -> None:
    q = _question(id="Q001")
    results = [_run("Q001", retrieved=[_retrieved(page=1, section="§1")])]
    assert mean_reciprocal_rank([q], results) == 0.0


def test_precision_at_k_averages_per_question() -> None:
    q = _question(id="Q001")
    results = [_run("Q001", retrieved=[_retrieved(page=40), _retrieved(page=1, section="§1")])]
    assert precision_at_k([q], results) == 0.5


def test_precision_at_k_excludes_empty_retrieval() -> None:
    q = _question(id="Q001")
    results = [_run("Q001", retrieved=[])]
    assert precision_at_k([q], results) is None


# --- citation_precision / hallucinated_citation_rate -----------------------


def test_citation_precision_global_ratio() -> None:
    q = _question(id="Q001")
    results = [_run("Q001", citations=[_citation(page=40), _citation(page=1, section="§1")])]
    assert citation_precision([q], results) == 0.5


def test_citation_precision_none_when_no_citations() -> None:
    q = _question(id="Q001")
    results = [_run("Q001", citations=[])]
    assert citation_precision([q], results) is None


def test_hallucinated_citation_rate_averages_across_answered() -> None:
    results = [_run("Q001", invented_citations=2), _run("Q002", invented_citations=0)]
    assert hallucinated_citation_rate(results) == 1.0


def test_hallucinated_citation_rate_none_when_all_errored() -> None:
    results = [_run("Q001", error="boom")]
    assert hallucinated_citation_rate(results) is None


# --- answer_accuracy ---------------------------------------------------


def test_answer_accuracy_layer1_pass_counts_correct() -> None:
    scored = [_scored("Q001", points_missing=[], layer2=JudgeResult(score=0, rationale="x"))]
    assert answer_accuracy(scored) == 1.0


def test_answer_accuracy_layer1_fail_layer2_full_score_counts_correct() -> None:
    scored = [_scored("Q001", points_missing=["x"], layer2=JudgeResult(score=2, rationale="ok"))]
    assert answer_accuracy(scored) == 1.0


def test_answer_accuracy_layer1_fail_layer2_partial_does_not_count() -> None:
    scored = [_scored("Q001", points_missing=["x"], layer2=JudgeResult(score=1, rationale="ok"))]
    assert answer_accuracy(scored) == 0.0


def test_answer_accuracy_layer1_fail_layer2_error_does_not_count() -> None:
    scored = [
        _scored("Q001", points_missing=["x"], layer2=JudgeError(raw_response="", error="bad"))
    ]
    assert answer_accuracy(scored) == 0.0


def test_answer_accuracy_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        answer_accuracy([])


# --- abstention rates ----------------------------------------------------


def test_correct_abstention_rate() -> None:
    should_abstain = _question(id="Q001", gold_sources=[], should_abstain=True)
    results = [_run("Q001", abstained=True)]
    assert correct_abstention_rate([should_abstain], results) == 1.0


def test_false_abstention_rate() -> None:
    answerable = _question(id="Q001")
    results = [_run("Q001", abstained=True)]
    assert false_abstention_rate([answerable], results) == 1.0


def test_false_abstention_rate_zero_when_answered() -> None:
    answerable = _question(id="Q001")
    results = [_run("Q001", abstained=False)]
    assert false_abstention_rate([answerable], results) == 0.0


def test_abstention_rates_none_when_no_matching_questions() -> None:
    answerable = _question(id="Q001")
    results = [_run("Q001", abstained=False)]
    assert correct_abstention_rate([answerable], results) is None


# --- latency / cost ---------------------------------------------------


def test_latency_percentiles_p50_p95() -> None:
    results = [
        _run("Q001", total_ms=1.0, retrieval_ms=1.0, generation_ms=0.0),
        _run("Q002", total_ms=2.0, retrieval_ms=2.0, generation_ms=0.0),
        _run("Q003", total_ms=3.0, retrieval_ms=3.0, generation_ms=0.0),
        _run("Q004", total_ms=4.0, retrieval_ms=4.0, generation_ms=0.0),
        _run("Q005", total_ms=5.0, retrieval_ms=5.0, generation_ms=0.0),
    ]
    p50, p95 = latency_percentiles(results)
    assert p50["total_ms"] == 3.0
    assert p95["total_ms"] == pytest.approx(4.8)


def test_latency_percentiles_excludes_errored() -> None:
    results = [_run("Q001", total_ms=10.0), _run("Q002", error="boom", total_ms=0.0)]
    p50, _ = latency_percentiles(results)
    assert p50["total_ms"] == 10.0


def test_latency_percentiles_raises_when_all_errored() -> None:
    with pytest.raises(ValueError, match="at least one"):
        latency_percentiles([_run("Q001", error="boom")])


def test_cost_per_query_avg_skips_none() -> None:
    results = [_run("Q001", cost_usd=0.002), _run("Q002", cost_usd=None)]
    assert cost_per_query_avg(results) == 0.002


def test_cost_per_query_avg_none_when_all_missing() -> None:
    assert cost_per_query_avg([_run("Q001", cost_usd=None)]) is None


# --- compute_metrics (integration) ----------------------------------------


def _config() -> EvalConfig:
    return EvalConfig(
        chunking_strategy="structural",
        reranker="cross_encoder",
        embedding_provider="e5_local",
        llm_provider="fake",
        llm_model="fake",
    )


def test_compute_metrics_end_to_end_round_trips_as_json() -> None:
    questions = [_question(id="Q001"), _question(id="Q002", gold_sources=[], should_abstain=True)]
    report = EvalReport(
        config=_config(),
        config_hash="abc123",
        created_at=datetime.now(UTC),
        n_questions=2,
        results=[
            _run("Q001", retrieved=[_retrieved()], citations=[_citation()]),
            _run("Q002", abstained=True, retrieved=[], citations=[]),
        ],
    )
    scored = ScoredReport(
        config_hash="abc123",
        scored_at=datetime.now(UTC),
        results=[_scored("Q001"), _scored("Q002")],
    )

    metrics = compute_metrics(report, scored, questions)

    assert metrics.config_hash == "abc123"
    assert metrics.n_questions == 2
    assert metrics.n_errors == 0
    assert metrics.recall_at_k == 1.0
    assert metrics.correct_abstention_rate == 1.0

    round_tripped = EvalMetrics.model_validate_json(metrics.model_dump_json())
    assert round_tripped == metrics


def test_compute_metrics_raises_on_empty_report() -> None:
    report = EvalReport(
        config=_config(),
        config_hash="abc123",
        created_at=datetime.now(UTC),
        n_questions=0,
        results=[],
    )
    scored = ScoredReport(config_hash="abc123", scored_at=datetime.now(UTC), results=[])
    with pytest.raises(ValueError, match="no results"):
        compute_metrics(report, scored, [])
