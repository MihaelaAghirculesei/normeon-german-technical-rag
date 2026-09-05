"""A deterministic ``LlmClient`` for tests and offline runs
(``llm_provider = "fake"``).

It does not call a model. It echoes a fixed German sentence that cites
whatever ``[S1]``, ``[S2]``, ... markers appear in the *context block* of
the rendered prompt, or returns exactly ``NICHT_GEFUNDEN`` when none do --
which is what the real prompt asks the model to do with an empty or
useless context. Enough to exercise the context builder, the endpoint,
and citation validation (Day 12) without a network call or a key.

It only reads between the prompt's ``Quellen:`` and ``Frage:`` markers
(falling back to the whole message if it can't find them, so a bare test
string still works). Earlier this scanned the *whole* user message,
which meant the literal ``[S1]``/``[S2]`` in the prompt's own rule text
made it "cite" a marker that was not actually offered whenever the real
context held zero or exactly one source -- exactly what citation
validation (``domain/citations.extract_and_validate``) exists to catch,
so it needed fixing at the source, not worked around per test.
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator

from app.adapters.llm.base import LlmResponse

_MARKER = re.compile(r"\[S\d+\]")
_CONTEXT_START = "Quellen:"
_CONTEXT_END = "\n\nFrage:"
_STREAM_CHUNK_CHARS = 8


def _context_block(user: str) -> str:
    start = user.find(_CONTEXT_START)
    if start == -1:
        return user
    start += len(_CONTEXT_START)
    end = user.find(_CONTEXT_END, start)
    return user[start:] if end == -1 else user[start:end]


class FakeLlmClient:
    name = "fake"

    def __init__(self, canned: str | None = None) -> None:
        self._canned = canned

    async def complete(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> LlmResponse:
        if self._canned is not None:
            return LlmResponse(text=self._canned, model=self.name)

        markers = list(dict.fromkeys(_MARKER.findall(_context_block(user))))
        if not markers:
            return LlmResponse(text="NICHT_GEFUNDEN", model=self.name)

        cited = " ".join(markers)
        text = (
            f"Nach den vorliegenden Quellen {cited} gelten die dort genannten "
            f"Anforderungen. {markers[0]}"
        )
        return LlmResponse(text=text, model=self.name)

    async def stream(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[str]:
        """Fixed-size character chunks of the same reply `complete` would
        give -- deterministic and trivially reassembled (`"".join(chunks)
        == complete(...).text`), enough to exercise the streaming endpoint
        without a network call."""
        text = (
            await self.complete(
                system=system, user=user, temperature=temperature, max_tokens=max_tokens
            )
        ).text
        for i in range(0, len(text), _STREAM_CHUNK_CHARS):
            yield text[i : i + _STREAM_CHUNK_CHARS]
