"""The generation seam.

Day 11 needs exactly one thing from an LLM: given a system instruction
and a user message, return the completion text (and, if the server
reports them, the token counts). The multi-provider Protocol, timeouts
and retries, and cost tracking are Day 13 / Day 15 -- this stays
deliberately small so those days have something to widen rather than
rewrite.

``complete`` is ``async`` because the real implementations are network
calls; the fake is ``async`` too so callers never branch on which one
they hold.
"""

from __future__ import annotations

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
