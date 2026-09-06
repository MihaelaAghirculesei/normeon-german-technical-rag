"""Unit tests for services/pricing.py. `load_pricing` reads the real
`backend/pricing.yaml` (no mocking the file -- this is what the "Fatto
quando" bar cares about: the numbers in that file are actually usable).
"""

from app.services.pricing import calculate_cost, load_pricing


def test_load_pricing_reads_the_real_file() -> None:
    pricing = load_pricing()

    assert "gpt-4o-mini" in pricing
    entry = pricing["gpt-4o-mini"]
    assert entry.input_per_mtok == 0.15
    assert entry.output_per_mtok == 0.60


def test_load_pricing_is_cached() -> None:
    assert load_pricing() is load_pricing()


def test_calculate_cost_matches_the_per_mtok_math() -> None:
    cost = calculate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)

    assert cost == round(1000 / 1_000_000 * 0.15 + 500 / 1_000_000 * 0.60, 6)


def test_calculate_cost_is_none_for_an_unpriced_model() -> None:
    assert calculate_cost("some-self-hosted-model", 100, 50) is None


def test_calculate_cost_is_none_when_prompt_tokens_is_unknown() -> None:
    assert calculate_cost("gpt-4o-mini", None, 500) is None


def test_calculate_cost_is_none_when_completion_tokens_is_unknown() -> None:
    assert calculate_cost("gpt-4o-mini", 1000, None) is None


def test_calculate_cost_zero_tokens_is_zero_cost_not_none() -> None:
    assert calculate_cost("gpt-4o-mini", 0, 0) == 0.0
