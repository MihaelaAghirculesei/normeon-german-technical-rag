"""Pure tests for `rrf` (Reciprocal Rank Fusion).

No database, no retrieval -- these pin the maths and the tie-breaking on
hand-built rankings of string ids.
"""

import pytest

from app.domain.fusion import rrf


def _scores(fused: list[tuple[str, float]]) -> dict[str, float]:
    return dict(fused)


def _order(fused: list[tuple[str, float]]) -> list[str]:
    return [doc_id for doc_id, _ in fused]


def test_single_ranking_keeps_its_order() -> None:
    assert _order(rrf([["a", "b", "c"]])) == ["a", "b", "c"]


def test_single_ranking_scores_are_one_over_k_plus_rank() -> None:
    fused = _scores(rrf([["a", "b", "c"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 63)


def test_two_identical_rankings_double_every_score() -> None:
    one = _scores(rrf([["a", "b", "c"]]))
    two = _scores(rrf([["a", "b", "c"], ["a", "b", "c"]]))
    for doc_id in one:
        assert two[doc_id] == pytest.approx(2 * one[doc_id])


def test_consensus_at_rank_two_beats_a_lone_rank_one() -> None:
    # `x` is #1 in one ranking only; `a` is #2 in both. For any k > 0,
    # 2/(k+2) > 1/(k+1), so the agreed item wins.
    fused = rrf([["x", "a"], ["y", "a"]])
    assert _order(fused)[0] == "a"


def test_disjoint_rankings_keep_all_items() -> None:
    fused = rrf([["a", "b"], ["c", "d"]])
    assert set(_order(fused)) == {"a", "b", "c", "d"}


def test_an_empty_ranking_contributes_nothing() -> None:
    with_empty = rrf([["a", "b", "c"], []])
    without = rrf([["a", "b", "c"]])
    assert _scores(with_empty) == pytest.approx(_scores(without))


def test_all_empty_rankings_return_empty() -> None:
    assert rrf([[], [], []]) == []


def test_no_rankings_at_all_return_empty() -> None:
    assert rrf([]) == []


def test_weights_scale_a_branch_contribution() -> None:
    base = _scores(rrf([["a", "b"], ["c", "d"]], weights=[1.0, 1.0]))
    heavy = _scores(rrf([["a", "b"], ["c", "d"]], weights=[3.0, 1.0]))
    assert heavy["a"] == pytest.approx(3 * base["a"])
    assert heavy["c"] == pytest.approx(base["c"])


def test_zero_weight_drops_a_ranking() -> None:
    fused = _scores(rrf([["a", "b", "c"], ["z"]], weights=[1.0, 0.0]))
    assert "z" not in fused or fused["z"] == pytest.approx(0.0)
    assert fused["a"] == pytest.approx(1 / 61)


def test_a_weighted_ranking_can_overturn_the_order() -> None:
    # `b` trails `a` on equal weights; a heavy second ranking that favours
    # `b` flips them.
    equal = _order(rrf([["a", "b"], ["b", "a"]], weights=[1.0, 1.0]))
    assert equal[0] == "a"  # tie broken by first appearance
    tilted = _order(rrf([["a", "b"], ["b", "a"]], weights=[1.0, 5.0]))
    assert tilted[0] == "b"


def test_weights_wrong_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="one per ranking"):
        rrf([["a"], ["b"]], weights=[1.0])


@pytest.mark.parametrize("bad_k", [0, -1, -60])
def test_non_positive_k_is_rejected(bad_k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        rrf([["a", "b"]], k=bad_k)


def test_smaller_k_sharpens_the_gap_between_ranks() -> None:
    flat = _scores(rrf([["a", "b"]], k=1000))
    steep = _scores(rrf([["a", "b"]], k=1))
    assert steep["a"] - steep["b"] > flat["a"] - flat["b"]


def test_a_repeated_id_within_one_ranking_counts_once_at_its_best_rank() -> None:
    fused = _scores(rrf([["a", "b", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)  # rank 1, not rank 1 + rank 3


def test_ties_break_by_first_appearance_across_rankings() -> None:
    # `p` and `q` are symmetric (each #1 once, #2 once) -> equal score.
    # `p` appears first in ranking 0, so it comes first.
    fused = rrf([["p", "q"], ["q", "p"]])
    assert _order(fused) == ["p", "q"]
    scores = _scores(fused)
    assert scores["p"] == pytest.approx(scores["q"])


def test_ranking_order_of_inputs_does_not_change_scores() -> None:
    a = _scores(rrf([["a", "b", "c"], ["c", "b"]]))
    b = _scores(rrf([["c", "b"], ["a", "b", "c"]]))
    assert a == pytest.approx(b)


def test_output_is_sorted_by_score_descending() -> None:
    fused = rrf([["a", "b", "c", "d"], ["c", "a", "e"]])
    values = [score for _, score in fused]
    assert values == sorted(values, reverse=True)


def test_earlier_rank_always_contributes_more() -> None:
    fused = _scores(rrf([["a", "b", "c", "d", "e"]]))
    ordered = [fused[x] for x in ("a", "b", "c", "d", "e")]
    assert ordered == sorted(ordered, reverse=True)


def test_three_branches_hand_computed() -> None:
    # vector: [a, b], fts: [b, c], trgm: [c]
    fused = _scores(rrf([["a", "b"], ["b", "c"], ["c"]], k=10))
    assert fused["a"] == pytest.approx(1 / 11)
    assert fused["b"] == pytest.approx(1 / 12 + 1 / 11)
    assert fused["c"] == pytest.approx(1 / 12 + 1 / 11)


def test_integer_weights_are_accepted() -> None:
    fused = _scores(rrf([["a"], ["b"]], weights=[2, 1]))
    assert fused["a"] == pytest.approx(2 / 61)
    assert fused["b"] == pytest.approx(1 / 61)


def test_result_items_are_str_float_pairs() -> None:
    fused = rrf([["a", "b"]])
    assert all(isinstance(k, str) and isinstance(v, float) for k, v in fused)


def test_fusion_can_surface_a_chunk_that_leads_no_single_ranking() -> None:
    # `c` is #2 in every ranking and #1 in none; each branch leads with a
    # different item. Consensus lifts `c` to the top of the fused list --
    # the property Day 8 is really after.
    fused = rrf([["a", "c"], ["b", "c"], ["d", "c"]])
    assert _order(fused)[0] == "c"


def test_large_input_stays_consistent_and_complete() -> None:
    r1 = [f"d{i}" for i in range(100)]
    r2 = [f"d{i}" for i in range(50, 150)]
    fused = rrf([r1, r2])
    assert len(fused) == 150
    assert _order(fused)[0] == "d50"  # only id that is rank 1 in r2 and top-tier in r1
