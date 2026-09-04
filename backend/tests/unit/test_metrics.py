"""Unit tests for the minimal in-process Counter (core/metrics.py)."""

import pytest

from app.core.metrics import Counter, hallucinated_citation_total


def test_counter_starts_at_zero() -> None:
    counter = Counter("test_counter")

    assert counter.value == 0


def test_counter_increments_by_one_by_default() -> None:
    counter = Counter("test_counter")

    counter.inc()
    counter.inc()

    assert counter.value == 2


def test_counter_increments_by_the_given_amount() -> None:
    counter = Counter("test_counter")

    counter.inc(3)
    counter.inc(4)

    assert counter.value == 7


def test_counter_rejects_a_negative_increment() -> None:
    counter = Counter("test_counter")

    with pytest.raises(ValueError, match="amount must be >= 0"):
        counter.inc(-1)


def test_counters_are_independent_instances() -> None:
    a, b = Counter("a"), Counter("b")

    a.inc(5)

    assert a.value == 5
    assert b.value == 0


def test_hallucinated_citation_total_is_a_shared_counter() -> None:
    # A module-level singleton, so this only checks its identity and
    # interface, not a specific value -- other tests in the same process
    # increment it too.
    before = hallucinated_citation_total.value

    hallucinated_citation_total.inc()

    assert hallucinated_citation_total.value == before + 1
