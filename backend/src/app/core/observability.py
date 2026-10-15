"""Tracing, and the phase timer that feeds both traces and metrics
(Giorno 27).

`phase("rerank")` opens an OpenTelemetry span *and* records the duration in
`normeon_phase_seconds{phase="rerank"}`, so traces and the latency metrics
cannot disagree about where the time went. The phases are `parse`, `embed`,
`retrieve.vector`, `retrieve.fts`, `retrieve.trgm`, `fuse`, `rerank` and
`generate`, nested under the HTTP request's span.

Where traces go (`settings.otel_exporter`):
  - `console` (development): one readable `trace.completed` log line per
    request, with every phase indented under its parent and its duration
    -- the plan's "a request produces a readable trace with the time of
    each phase";
  - `otlp` (production): batched to a collector at `settings.otel_endpoint`;
  - `none`: tracing off.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.core.config import settings
from app.core.metrics import PHASE_SECONDS

REQUEST_ID_ATTRIBUTE = "normeon.request_id"

_tracer = trace.get_tracer("normeon")
_log = structlog.get_logger(__name__)


@contextmanager
def phase(name: str, **attributes: str | int | float | bool) -> Iterator[trace.Span]:
    """Time one pipeline phase as a span and as a latency observation."""
    started = time.perf_counter()
    with _tracer.start_as_current_span(name, attributes=attributes) as span:
        try:
            yield span
        finally:
            PHASE_SECONDS.labels(phase=name).observe(time.perf_counter() - started)


class PhaseTimer:
    """`phase` for work that is not one block of code -- the streamed
    generation, which yields to its caller between tokens. The span is
    started but not made current, so nothing is attached to a context the
    generator might be resumed from somewhere else."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._started = time.perf_counter()
        self._span = _tracer.start_span(name)

    def end(self) -> None:
        PHASE_SECONDS.labels(phase=self.name).observe(time.perf_counter() - self._started)
        self._span.end()


class TraceSummary(SpanProcessor):
    """Collects the spans of each trace and, when its root span ends, logs
    the whole trace as one indented list of phases with durations. Traces
    whose root never ends are dropped once `max_traces` are pending."""

    def __init__(self, max_traces: int = 1000) -> None:
        self._pending: OrderedDict[int, list[ReadableSpan]] = OrderedDict()
        self._max_traces = max_traces

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        if span.context is None:
            return
        trace_id = span.context.trace_id
        self._pending.setdefault(trace_id, []).append(span)
        if span.parent is not None and not span.parent.is_remote:
            if len(self._pending) > self._max_traces:
                self._pending.popitem(last=False)
            return
        spans = self._pending.pop(trace_id, [])
        attributes = span.attributes or {}
        _log.info(
            "trace.completed",
            trace_id=format(trace_id, "032x"),
            request_id=attributes.get(REQUEST_ID_ATTRIBUTE),
            root=span.name,
            total_ms=_ms(span),
            phases=summarise(spans),
        )

    def shutdown(self) -> None:
        self._pending.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def summarise(spans: list[ReadableSpan]) -> list[str]:
    """`["POST /api/v1/chat 812.4 ms", "  retrieve 640.1 ms", ...]`: every
    span in start order, indented by its depth in the trace."""
    by_id = {s.context.span_id: s for s in spans if s.context is not None}

    def depth(span: ReadableSpan) -> int:
        level = 0
        parent = span.parent
        while parent is not None and parent.span_id in by_id:
            level += 1
            parent = by_id[parent.span_id].parent
        return level

    ordered = sorted(spans, key=lambda s: s.start_time or 0)
    return [f"{'  ' * depth(s)}{s.name} {_ms(s)} ms" for s in ordered]


def _ms(span: ReadableSpan) -> float:
    if span.start_time is None or span.end_time is None:
        return 0.0
    return round((span.end_time - span.start_time) / 1e6, 1)


def configure_tracing(app: FastAPI) -> None:
    """Install the tracer provider for `settings.otel_exporter` and trace
    every HTTP request (health and metrics excluded)."""
    if settings.otel_exporter == "none":
        return
    provider = TracerProvider(resource=Resource.create({"service.name": settings.app_name}))
    if settings.otel_exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint))
        )
    else:
        provider.add_span_processor(TraceSummary())
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=provider,
        excluded_urls="health,metrics",
        # One span per ASGI message would mean one per streamed token.
        exclude_spans=["receive", "send"],
    )


def tag_current_span(**attributes: Any) -> None:
    """Attach attributes to the span being recorded, if any."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attributes(attributes)
