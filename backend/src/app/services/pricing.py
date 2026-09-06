"""Cost tracking (plan, Giorno 15): a versioned per-model pricing table
(``backend/pricing.yaml``) plus the pure calculation that turns
``(model, prompt_tokens, completion_tokens)`` into a USD cost.

Mirrors the ``services/prompts.py`` pattern: a cached loader reading a
file that lives outside ``src/app`` (same packaging caveat -- fine for
source/docker/tests, revisit if this app is ever ``pip install``ed).

Looked up by the *exact* model id string a response resolves to -- see
``pricing.yaml``'s own header for what that string is per provider. An
unpriced model is a normal, expected state (a self-hosted deployment
with a homegrown model id, or the ``fake`` provider), not an error, so
``calculate_cost`` returns ``None`` rather than fabricating a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_PRICING_FILE = Path(__file__).parents[3] / "pricing.yaml"


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float


@lru_cache
def load_pricing() -> dict[str, ModelPricing]:
    """Read and parse ``pricing.yaml``. Cached: the price table is small
    and fixed for the life of the process."""
    raw = yaml.safe_load(_PRICING_FILE.read_text(encoding="utf-8")) or {}
    models: dict[str, object] = raw.get("models") or {}
    return {
        str(model_id): ModelPricing(
            input_per_mtok=float(prices["input_per_mtok"]),  # type: ignore[index]
            output_per_mtok=float(prices["output_per_mtok"]),  # type: ignore[index]
        )
        for model_id, prices in models.items()
    }


def calculate_cost(
    model: str, prompt_tokens: int | None, completion_tokens: int | None
) -> float | None:
    """USD cost of one completion, or ``None`` if `model` isn't in
    ``pricing.yaml``, or either token count is unknown (some providers,
    and most streaming responses, don't report usage)."""
    if prompt_tokens is None or completion_tokens is None:
        return None
    pricing = load_pricing().get(model)
    if pricing is None:
        return None
    cost = (prompt_tokens / 1_000_000) * pricing.input_per_mtok
    cost += (completion_tokens / 1_000_000) * pricing.output_per_mtok
    return round(cost, 6)
