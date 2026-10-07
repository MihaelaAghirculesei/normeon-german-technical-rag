"""Bootstrap confidence intervals for the eval metrics.

Fifty questions -- ten per category -- is a small sample: one question
is two points of overall accuracy and ten points inside a category. A
bare "configuration B wins by 4 points" means nothing until it is set
against how much the number would move under a different draw of
questions. The percentile bootstrap answers that without distributional
assumptions, and `paired_bootstrap_diff` resamples *questions* (not two
independent samples), because every configuration answers the same
set: the paired comparison cancels the per-question difficulty that two
independent intervals would double count.

Pure and deterministic -- a fixed seed, so a report regenerated from the
same files prints the same intervals.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 20


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _percentile_interval(samples: list[float], confidence: float) -> tuple[float, float]:
    samples.sort()
    tail = (1 - confidence) / 2
    lo = samples[int(tail * (len(samples) - 1))]
    hi = samples[int(round((1 - tail) * (len(samples) - 1)))]
    return lo, hi


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float] | None:
    """Percentile-bootstrap interval for the mean of `values` (e.g. one
    0/1 per question for an accuracy). `None` on an empty sample."""
    if not values:
        return None
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    n = len(values)
    means = [_mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples)]
    return _percentile_interval(means, confidence)


def paired_bootstrap_diff(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float] | None:
    """Mean of `candidate - baseline` over paired observations (the same
    question under two configurations) and its percentile-bootstrap
    interval, as `(diff, lo, hi)`. An interval that excludes 0 is a
    difference this sample actually supports. `None` on an empty
    sample."""
    if len(baseline) != len(candidate):
        raise ValueError("paired samples must have the same length")
    if not baseline:
        return None
    diffs = [c - b for b, c in zip(baseline, candidate, strict=True)]
    ci = bootstrap_ci(diffs, confidence=confidence, n_resamples=n_resamples, seed=seed)
    assert ci is not None
    return _mean(diffs), ci[0], ci[1]
