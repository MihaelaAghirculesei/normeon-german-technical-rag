"""The generation seam.

Day 11 needs exactly one thing from an LLM: given a system instruction
and a user message, return the completion text (and, if the server
reports them, the token counts). Day 13 widened it with retries and
typed errors on `complete`, kept inside the adapters, not the Protocol.
Day 14 added `stream`, for `POST /api/v1/chat/stream`. Day 15 widens
`stream` from raw text pieces to `Delta` -- the plan's own sketch for
this day already shows `stream(...) -> AsyncIterator[Delta]` -- so a
streamed answer's cost can be tracked the same as a non-streamed one's,
without which `Delta.prompt_tokens`/`completion_tokens` would have
nowhere to travel from the wire to `services.generation.
generate_answer_stream`'s `query_logs` row.

``complete`` is ``async`` because the real implementations are network
calls; the fake is ``async`` too so callers never branch on which one
they hold.

``stream`` is declared as a plain (non-``async``) method returning
``AsyncGenerator[Delta, None]`` -- an async *generator function* (``async
def stream(...): yield ...``), when called, returns that iterator
directly without needing to be awaited first; declaring it ``async def
... -> AsyncGenerator[Delta, None]`` in the Protocol would instead
describe a coroutine that, once awaited, *returns* an iterator, which is
one indirection too many and does not match how an async generator
function actually behaves when called.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LlmResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Delta:
    """One piece of a streamed reply. `text` is usually the only
    populated field, repeated many times; `model`/`prompt_tokens`/
    `completion_tokens` show up on however many deltas the server
    chooses to report them on (for OpenAI-compatible servers that
    support `stream_options.include_usage`, typically only the final
    delta, which may carry no `text` at all)."""

    text: str = ""
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LlmClient(Protocol):
    name: str

    async def complete(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> LlmResponse: ...

    def stream(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[Delta]: ...
