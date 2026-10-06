"""Unit tests for app/eval/stats: the bootstrap intervals the matrix
report puts next to every headline number."""

import pytest

from app.eval.stats import bootstrap_ci, paired_bootstrap_diff


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = [1.0] * 35 + [0.0] * 15  # 0.70 accuracy over 50 questions

    ci = bootstrap_ci(values)

    assert ci is not None
    lo, hi = ci
    assert lo < 0.70 < hi
    # binomial standard error at n=50 is ~0.065, so the 95% interval is
    # roughly +/-0.13 -- the honest width of a 50-question eval
    assert 0.18 < hi - lo < 0.32


def test_bootstrap_ci_is_deterministic_for_a_seed() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert bootstrap_ci(values) == bootstrap_ci(values)
    assert bootstrap_ci(values, seed=1) == bootstrap_ci(values, seed=1)


def test_bootstrap_ci_of_a_constant_is_that_constant() -> None:
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)


def test_bootstrap_ci_empty_is_none() -> None:
    assert bootstrap_ci([]) is None


def test_bootstrap_ci_rejects_a_bad_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0], confidence=1.0)


def test_paired_diff_detects_a_consistent_improvement() -> None:
    # the candidate fixes 12 of the baseline's misses and breaks none
    baseline = [1.0] * 30 + [0.0] * 20
    candidate = [1.0] * 42 + [0.0] * 8

    result = paired_bootstrap_diff(baseline, candidate)

    assert result is not None
    diff, lo, hi = result
    assert diff == pytest.approx(0.24)
    assert lo > 0  # the interval excludes zero: a real difference


def test_paired_diff_does_not_call_noise_a_win() -> None:
    # two flips each way plus one net: +2 points, well inside the noise
    baseline = [1.0, 1.0, 0.0, 0.0, 0.0] + [1.0] * 30 + [0.0] * 15
    candidate = [0.0, 0.0, 1.0, 1.0, 1.0] + [1.0] * 30 + [0.0] * 15

    result = paired_bootstrap_diff(baseline, candidate)

    assert result is not None
    diff, lo, hi = result
    assert diff == pytest.approx(0.02)
    assert lo < 0 < hi


def test_paired_diff_requires_equal_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_diff([1.0], [1.0, 0.0])


def test_paired_diff_empty_is_none() -> None:
    assert paired_bootstrap_diff([], []) is None
