"""Retrieval, citation and outcome metrics computed over a real eval run
(plan, Giorno 19).

Pure: takes an already-produced `EvalReport` (Giorno 16), an already-
produced `ScoredReport` (Giorno 18) and the question set that drove
both, and returns one `EvalMetrics`. No I/O, no DB, no LLM -- everything
here is a re-read of data those two earlier stages already collected.

Gold-source matching (`recall_at_k`/`mrr`/`precision_at_k`/
`citation_precision`) is deliberately tolerant, per the plan's own
operational definition: same `document`, and *either* a section-string
overlap *or* the page within +/-1 -- not an exact match on both, since
chunk boundaries rarely land on the same page/section split a human
would draw by hand.

`answer_accuracy` operationalizes "Layer 1 + Layer 2" as an escalation,
not an average: Layer 1's strict substring check is authoritative when
it passes (cheap, unambiguous); when it fails -- often just a phrasing
difference from `expected_answer_points`, not a wrong answer -- Layer
2's judge score is consulted as the semantic fallback, and only a full
score of 2 counts as correct. A judge score of 1 (partially correct) is
Layer 1 disagreeing with a judge that itself isn't fully convinced, so
it doesn't flip a Layer-1 failure to a pass.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from app.eval.models import CitationRecord, EvalQuestion, EvalReport, QuestionRun, RetrievedRecord
from app.eval.scoring import JudgeResult, ScoredQuestionRun, ScoredReport

GoldMatchable = RetrievedRecord | CitationRecord


def _matches_gold(
    question: EvalQuestion, *, filename: str, page_from: int, page_to: int, section_path: str | None
) -> bool:
    for gold in question.gold_sources:
        if gold.document != filename:
            continue
        if gold.section is None and gold.page is None:
            return True
        section_hit = (
            gold.section is not None
            and section_path is not None
            and (gold.section in section_path or section_path in gold.section)
        )
        page_hit = gold.page is not None and page_from - 1 <= gold.page <= page_to + 1
        if section_hit or page_hit:
            return True
    return False


def _any_gold_match(question: EvalQuestion, record: GoldMatchable) -> bool:
    return _matches_gold(
        question,
        filename=record.filename,
        page_from=record.page_from,
        page_to=record.page_to,
        section_path=record.section_path,
    )


def _gold_bearing_runs(
    questions: list[EvalQuestion], results: list[QuestionRun]
) -> list[tuple[EvalQuestion, QuestionRun]]:
    """Only questions with at least one gold source, matched to a run
    that actually completed -- an `unanswerable` question has no gold
    source to recall, and an errored run has no retrieval to score."""
    by_id = {r.question_id: r for r in results}
    pairs = []
    for question in questions:
        if not question.gold_sources:
            continue
        run = by_id.get(question.id)
        if run is None or run.error is not None:
            continue
        pairs.append((question, run))
    return pairs


def recall_at_k(questions: list[EvalQuestion], results: list[QuestionRun]) -> float | None:
    """Fraction of gold-bearing questions where at least one gold source
    is among the chunks that actually made it into context."""
    pairs = _gold_bearing_runs(questions, results)
    if not pairs:
        return None
    hits = [any(_any_gold_match(q, r) for r in run.retrieved) for q, run in pairs]
    return sum(hits) / len(hits)


def mean_reciprocal_rank(questions: list[EvalQuestion], results: list[QuestionRun]) -> float | None:
    pairs = _gold_bearing_runs(questions, results)
    if not pairs:
        return None
    reciprocal_ranks = []
    for question, run in pairs:
        rank = next(
            (i + 1 for i, r in enumerate(run.retrieved) if _any_gold_match(question, r)), None
        )
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def precision_at_k(questions: list[EvalQuestion], results: list[QuestionRun]) -> float | None:
    """Averaged per question, over questions that retrieved at least one
    chunk -- an empty retrieval has undefined (0/0) precision, not 0."""
    pairs = _gold_bearing_runs(questions, results)
    precisions = [
        sum(_any_gold_match(q, r) for r in run.retrieved) / len(run.retrieved)
        for q, run in pairs
        if run.retrieved
    ]
    if not precisions:
        return None
    return sum(precisions) / len(precisions)


def citation_precision(questions: list[EvalQuestion], results: list[QuestionRun]) -> float | None:
    """Fraction of all valid citations, across the whole run, that point
    at a gold source -- a single global ratio, not averaged per question,
    so a question with many citations isn't diluted to the same weight
    as one with a single lucky hit."""
    by_id = {q.id: q for q in questions}
    total = 0
    matched = 0
    for run in results:
        if run.error is not None:
            continue
        question = by_id.get(run.question_id)
        if question is None:
            continue
        for citation in run.citations:
            total += 1
            if _any_gold_match(question, citation):
                matched += 1
    if total == 0:
        return None
    return matched / total


def hallucinated_citation_rate(results: list[QuestionRun]) -> float | None:
    """Invented `[S..]` markers (dropped before the answer reached the
    user, see `domain.citations.extract_and_validate`) per answered
    question -- a model that frequently invents markers is a real risk
    even though none of them ever surfaced."""
    answered = [r for r in results if r.error is None]
    if not answered:
        return None
    return sum(r.invented_citations for r in answered) / len(answered)


def answer_accuracy(scored_results: list[ScoredQuestionRun]) -> float:
    if not scored_results:
        raise ValueError("need at least one scored question")
    correct = sum(
        1
        for r in scored_results
        if r.layer1.all_present or (isinstance(r.layer2, JudgeResult) and r.layer2.score == 2)
    )
    return correct / len(scored_results)


def correct_abstention_rate(
    questions: list[EvalQuestion], results: list[QuestionRun]
) -> float | None:
    """Among `should_abstain` questions with a completed run, the
    fraction that actually abstained."""
    by_id = {r.question_id: r for r in results}
    outcomes = [
        by_id[q.id].abstained
        for q in questions
        if q.should_abstain and q.id in by_id and by_id[q.id].error is None
    ]
    if not outcomes:
        return None
    return sum(outcomes) / len(outcomes)


def false_abstention_rate(
    questions: list[EvalQuestion], results: list[QuestionRun]
) -> float | None:
    """Among answerable questions with a completed run, the fraction
    that abstained anyway -- the cost of a confidence gate set too
    high."""
    by_id = {r.question_id: r for r in results}
    outcomes = [
        by_id[q.id].abstained
        for q in questions
        if not q.should_abstain and q.id in by_id and by_id[q.id].error is None
    ]
    if not outcomes:
        return None
    return sum(outcomes) / len(outcomes)


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (same convention as numpy's
    default) -- deterministic and dependency-free for the small samples
    a 50-question eval run produces."""
    if not values:
        raise ValueError("need at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def latency_percentiles(results: list[QuestionRun]) -> tuple[dict[str, float], dict[str, float]]:
    """p50/p95 for each phase and the total, over answered questions
    only -- an errored run's 0.0 placeholders would skew both ends."""
    answered = [r for r in results if r.error is None]
    if not answered:
        raise ValueError("need at least one non-errored question")
    phases = {
        "retrieval_ms": [r.retrieval_ms for r in answered],
        "generation_ms": [r.generation_ms for r in answered],
        "total_ms": [r.total_ms for r in answered],
    }
    p50 = {phase: _percentile(values, 50) for phase, values in phases.items()}
    p95 = {phase: _percentile(values, 95) for phase, values in phases.items()}
    return p50, p95


def cost_per_query_avg(results: list[QuestionRun]) -> float | None:
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    if not costs:
        return None
    return sum(costs) / len(costs)


class EvalMetrics(BaseModel):
    config_hash: str
    n_questions: int
    n_errors: int
    recall_at_k: float | None
    mrr: float | None
    precision_at_k: float | None
    citation_precision: float | None
    hallucinated_citation_rate: float | None
    answer_accuracy: float
    correct_abstention_rate: float | None
    false_abstention_rate: float | None
    latency_ms_p50: dict[str, float]
    latency_ms_p95: dict[str, float]
    cost_per_query_avg: float | None


def compute_metrics(
    report: EvalReport, scored: ScoredReport, questions: list[EvalQuestion]
) -> EvalMetrics:
    if not report.results:
        raise ValueError("report has no results")
    p50, p95 = latency_percentiles(report.results)
    return EvalMetrics(
        config_hash=report.config_hash,
        n_questions=len(report.results),
        n_errors=sum(1 for r in report.results if r.error is not None),
        recall_at_k=recall_at_k(questions, report.results),
        mrr=mean_reciprocal_rank(questions, report.results),
        precision_at_k=precision_at_k(questions, report.results),
        citation_precision=citation_precision(questions, report.results),
        hallucinated_citation_rate=hallucinated_citation_rate(report.results),
        answer_accuracy=answer_accuracy(scored.results),
        correct_abstention_rate=correct_abstention_rate(questions, report.results),
        false_abstention_rate=false_abstention_rate(questions, report.results),
        latency_ms_p50=p50,
        latency_ms_p95=p95,
        cost_per_query_avg=cost_per_query_avg(report.results),
    )
