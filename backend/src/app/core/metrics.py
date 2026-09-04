"""Minimal in-process counters.

No dependency: a real Prometheus endpoint is Day 27 (`/metrics`,
`docs/piano` "Osservabilita"). Until then this gives callers something
concrete to increment now (`hallucinated_citation_total`, Giorno 12)
without pulling in `prometheus_client` for a single counter. The
interface (`.inc()`) is the one `prometheus_client.Counter` exposes, so
swapping the implementation later is a one-line change per counter, not
a rewrite of call sites.
"""

from __future__ import annotations

from threading import Lock


class Counter:
    """A thread-safe, monotonically increasing counter."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._value = 0
        self._lock = Lock()

    def inc(self, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("Counter.inc() amount must be >= 0")
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        return self._value


hallucinated_citation_total = Counter("hallucinated_citation_total")
