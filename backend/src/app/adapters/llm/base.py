"""The generation seam.

Day 11 needs exactly one thing from an LLM: given a system instruction
and a user message, return the completion text (and, if the server
reports them, the token counts). Day 13 widened it with retries and
typed errors on `complete`, kept inside the adapters, not the Protocol.
Day 14 adds `stream`, for `POST /api/v1/chat/stream`. Cost tracking is
still Day 15 -- this stays deliberately small so that day has something
to widen rather than rewrite.

``complete`` is ``async`` because the real implementations are network
calls; the fake is ``async`` too so callers never branch on which one
they hold.

``stream`` is declared as a plain (non-``async``) method returning
``AsyncGenerator[str, None]`` -- an async *generator function* (``async def
stream(...): yield ...``), when called, returns that iterator directly
without needing to be awaited first; declaring it ``async def ... ->
AsyncGenerator[str, None]`` in the Protocol would instead describe a coroutine
that, once awaited, *returns* an iterator, which is one indirection too
many and does not match how an async generator function actually
behaves when called.
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


class LlmClient(Protocol):
    name: str

    async def complete(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> LlmResponse: ...

    def stream(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[str]: ...
