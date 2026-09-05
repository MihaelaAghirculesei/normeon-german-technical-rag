"""Unit tests for core/streaming.with_heartbeat.

Uses small real sleeps (asyncio_mode=auto, no fixture/marker -- matches
the rest of the async unit suite) since there's no fake clock in this
project; the intervals are short enough (tens of ms) to stay fast.
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from app.core.streaming import with_heartbeat

_HEARTBEAT = object()


async def _fast_source() -> AsyncGenerator[str]:
    yield "a"
    yield "b"


async def _slow_then_fast_source(delay: float) -> AsyncGenerator[str]:
    await asyncio.sleep(delay)
    yield "a"
    yield "b"


async def _raising_source() -> AsyncGenerator[str]:
    raise ValueError("boom")
    yield "unreachable"  # pragma: no cover -- makes this a generator function


async def _closable_source(closed: list[bool]) -> AsyncGenerator[str]:
    try:
        yield "a"
        await asyncio.sleep(10)
        yield "b"  # pragma: no cover -- never reached, closed first
    finally:
        closed.append(True)


async def test_a_fast_source_produces_no_heartbeats() -> None:
    wrapped = with_heartbeat(_fast_source(), interval=1.0, heartbeat=_HEARTBEAT)
    items = [item async for item in wrapped]

    assert items == ["a", "b"]


async def test_a_slow_source_is_heartbeated_without_losing_its_item() -> None:
    items = [
        item
        async for item in with_heartbeat(
            _slow_then_fast_source(0.12), interval=0.04, heartbeat=_HEARTBEAT
        )
    ]

    assert items[-2:] == ["a", "b"]
    assert items.count(_HEARTBEAT) >= 2


async def test_an_exception_from_the_source_propagates() -> None:
    with pytest.raises(ValueError, match="boom"):
        async for _ in with_heartbeat(_raising_source(), interval=1.0, heartbeat=_HEARTBEAT):
            pass


async def test_closing_the_wrapper_closes_the_wrapped_source() -> None:
    closed: list[bool] = []
    wrapped = with_heartbeat(_closable_source(closed), interval=1.0, heartbeat=_HEARTBEAT)

    async for _ in wrapped:
        break  # stop after the first item, while the source is still suspended

    await wrapped.aclose()

    assert closed == [True]


async def test_closing_the_wrapper_while_a_heartbeat_wait_is_pending_still_closes_the_source() -> (
    None
):
    closed: list[bool] = []
    wrapped = with_heartbeat(_closable_source(closed), interval=0.02, heartbeat=_HEARTBEAT)

    first = await wrapped.__anext__()
    assert first == "a"
    # The source is now asleep for 10s; the next __anext__ would just
    # heartbeat forever. Close without waiting for another tick.
    await wrapped.aclose()

    assert closed == [True]
