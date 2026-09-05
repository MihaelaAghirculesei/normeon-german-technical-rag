"""A generic async-iterator heartbeat wrapper (plan, Giorno 14).

A long gap between two events -- a slow retrieval, a model "thinking"
before its first token -- can trip an idle-connection timeout on an
intermediate proxy. `with_heartbeat` periodically yields a caller-supplied
sentinel when the wrapped iterator hasn't produced anything for
`interval` seconds, WITHOUT cancelling the pending fetch.

That last part is the reason this isn't just `asyncio.wait_for(agen.
__anext__(), interval)` in a loop: `wait_for` *cancels* the wrapped
coroutine the moment its timeout fires, which for a network read means
tearing down the connection just because the model was slow, not because
anything actually failed. `asyncio.wait` on the same pending `Task`,
re-awaited on the next tick if it timed out, does not have that problem
-- a timeout there means "not done yet," not "give up."
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator


async def with_heartbeat[T](
    source: AsyncGenerator[T], *, interval: float, heartbeat: T
) -> AsyncGenerator[T]:
    pending: asyncio.Task[T] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(source.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield heartbeat
                continue
            task, pending = pending, None
            try:
                yield task.result()
            except StopAsyncIteration:
                return
    finally:
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending
        await source.aclose()
