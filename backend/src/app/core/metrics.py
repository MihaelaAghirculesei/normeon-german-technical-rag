"""Prometheus metrics (Giorno 27), served at `/metrics`.

What an operator needs to see at a glance:
  - latency per pipeline phase (`normeon_phase_seconds`, recorded by
    `core.observability.phase` together with the trace span);
  - answers by outcome, so the abstention rate is a ratio of two series;
  - cumulative LLM cost;
  - citations the model invented, against the ones it got right;
  - LLM errors per provider and error code.

Process-local, like any Prometheus client: with several workers, the
scraper sums them. `hallucinated_citation_total` keeps the name and the
`.inc()` call sites it had as a plain counter since Giorno 12.
"""

from __future__ import annotations

from prometheus_client import REGISTRY, Counter, Histogram

PHASE_SECONDS = Histogram(
    "normeon_phase_seconds",
    "Time spent in one pipeline phase.",
    ["phase"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
ANSWERS_TOTAL = Counter(
    "normeon_answers_total",
    "Answers given, by outcome: answered, or abstained (NICHT_GEFUNDEN).",
    ["outcome"],
)
LLM_COST_USD_TOTAL = Counter(
    "normeon_llm_cost_usd_total",
    "Cumulative LLM cost in USD, for models with a known price.",
    ["model"],
)
CITATIONS_TOTAL = Counter(
    "normeon_citations_total",
    "Citation markers in answers that resolved to a retrieved source.",
)
hallucinated_citation_total = Counter(
    "normeon_hallucinated_citations_total",
    "Citation markers the model invented (no such source); dropped from answers.",
)
LLM_ERRORS_TOTAL = Counter(
    "normeon_llm_errors_total",
    "LLM calls that failed, by provider and error code.",
    ["provider", "code"],
)


def sample(name: str, labels: dict[str, str] | None = None) -> float:
    """Current value of one series, 0 if it has not been recorded yet --
    for tests and for the admin statistics."""
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0
