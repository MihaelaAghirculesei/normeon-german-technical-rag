import pytest

from app.eval.gate import GateQuestion, separability_auc, simulate_gate


def _q(
    qid: str, *, unanswerable: bool, top: float | None, abstained: bool, correct: bool
) -> GateQuestion:
    return GateQuestion(
        question_id=qid,
        should_abstain=unanswerable,
        top_score=top,
        abstained=abstained,
        correct=correct,
    )


RUN = [
    # answered correctly, high and low scores
    _q("Q1", unanswerable=False, top=0.95, abstained=False, correct=True),
    _q("Q2", unanswerable=False, top=0.55, abstained=False, correct=True),
    # wrongly abstained by the model, low score
    _q("Q3", unanswerable=False, top=0.20, abstained=True, correct=False),
    # unanswerable: one abstained, one answered (wrong)
    _q("Q4", unanswerable=True, top=0.10, abstained=True, correct=True),
    _q("Q5", unanswerable=True, top=0.40, abstained=False, correct=False),
]


def test_a_threshold_below_every_score_reproduces_the_run() -> None:
    point = simulate_gate(RUN, 0.0)

    assert point.answer_accuracy == pytest.approx(3 / 5)
    assert point.false_abstention_rate == pytest.approx(1 / 3)
    assert point.correct_abstention_rate == pytest.approx(1 / 2)
    assert point.llm_calls_saved == 0


def test_gating_an_unanswerable_question_makes_it_correct() -> None:
    point = simulate_gate(RUN, 0.5)

    # Q3, Q4, Q5 gated: Q5 flips to correct, Q3 stays wrong, Q4 stays right.
    assert point.answer_accuracy == pytest.approx(4 / 5)
    assert point.correct_abstention_rate == 1.0
    assert point.false_abstention_rate == pytest.approx(1 / 3)
    assert point.llm_calls_saved == 3


def test_gating_a_correct_answer_costs_it() -> None:
    point = simulate_gate(RUN, 0.6)

    assert point.answer_accuracy == pytest.approx(3 / 5)
    assert point.false_abstention_rate == pytest.approx(2 / 3)


def test_the_threshold_itself_passes() -> None:
    assert simulate_gate(RUN, 0.55).llm_calls_saved == 3


def test_an_empty_retrieval_is_gated_at_any_threshold() -> None:
    run = [*RUN, _q("Q6", unanswerable=False, top=None, abstained=True, correct=False)]

    assert simulate_gate(run, -1.0).llm_calls_saved == 1


def test_simulate_refuses_a_run_without_both_kinds_of_question() -> None:
    with pytest.raises(ValueError):
        simulate_gate([], 0.3)
    with pytest.raises(ValueError):
        simulate_gate(RUN[:3], 0.3)


def test_auc_is_one_when_a_threshold_separates_them() -> None:
    run = [
        _q("A", unanswerable=False, top=0.9, abstained=False, correct=True),
        _q("B", unanswerable=True, top=0.1, abstained=True, correct=True),
    ]
    assert separability_auc(run) == 1.0


def test_auc_counts_pairs_and_ties_half() -> None:
    # answerable 0.95, 0.55, 0.20 vs unanswerable 0.10, 0.40:
    # wins 2 + 2 + 1 = 5 of 6 pairs
    assert separability_auc(RUN) == pytest.approx(5 / 6)
    tie = [
        _q("A", unanswerable=False, top=0.5, abstained=False, correct=True),
        _q("B", unanswerable=True, top=0.5, abstained=True, correct=True),
    ]
    assert separability_auc(tie) == 0.5
