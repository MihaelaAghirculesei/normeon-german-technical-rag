"""Replay the pre-generation confidence gate over finished eval runs.

The gate skips the LLM when the top reranked chunk scores below
`min_rerank_score_for_answer`. Its effect on a run that already happened
is fully determined, question by question: below the threshold the
answer becomes NICHT_GEFUNDEN -- correct exactly when the question is
unanswerable -- and above it the recorded outcome stands. So a threshold
can be tuned from the reports alone, without new LLM calls.

The replay is only valid for runs the gate did not already filter: a
question it stopped never reached the model, so there is no recorded
outcome to fall back on above a lower threshold. Runs scored with a
threshold no top score fell below (any run with 0.0 and a cross-encoder,
whose scores are probabilities in [0, 1]) qualify.

Pure -- no DB, no files.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GateQuestion:
    """One question of a finished run: the top reranked score and what
    actually happened."""

    question_id: str
    should_abstain: bool
    top_score: float | None
    abstained: bool
    correct: bool


@dataclass(frozen=True, slots=True)
class GatePoint:
    threshold: float
    answer_accuracy: float
    false_abstention_rate: float
    correct_abstention_rate: float
    llm_calls_saved: int


def simulate_gate(questions: Sequence[GateQuestion], threshold: float) -> GatePoint:
    """The run's outcome metrics had the gate been set to `threshold`.
    A question with no retrieved chunk (`top_score` None) is gated at any
    threshold, as in `generate_answer`."""
    if not questions:
        raise ValueError("need at least one question")
    answerable = [q for q in questions if not q.should_abstain]
    unanswerable = [q for q in questions if q.should_abstain]
    if not answerable or not unanswerable:
        raise ValueError("need both answerable and unanswerable questions")

    def gated(q: GateQuestion) -> bool:
        return q.top_score is None or q.top_score < threshold

    def abstained(q: GateQuestion) -> bool:
        return gated(q) or q.abstained

    def correct(q: GateQuestion) -> bool:
        return q.should_abstain if gated(q) else q.correct

    return GatePoint(
        threshold=threshold,
        answer_accuracy=sum(correct(q) for q in questions) / len(questions),
        false_abstention_rate=sum(abstained(q) for q in answerable) / len(answerable),
        correct_abstention_rate=sum(abstained(q) for q in unanswerable) / len(unanswerable),
        llm_calls_saved=sum(gated(q) for q in questions),
    )


def separability_auc(questions: Sequence[GateQuestion]) -> float:
    """Probability that a random answerable question's top score beats a
    random unanswerable one's (ties count half): 1.0 means some threshold
    separates them perfectly, 0.5 means the score says nothing about it.
    Questions without a top score are left out."""
    pos = [q.top_score for q in questions if not q.should_abstain and q.top_score is not None]
    neg = [q.top_score for q in questions if q.should_abstain and q.top_score is not None]
    if not pos or not neg:
        raise ValueError("need scored answerable and unanswerable questions")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
