"""A deterministic ``LlmClient`` for tests and offline runs
(``llm_provider = "fake"``).

It does not call a model. It echoes a fixed German sentence that cites
whatever ``[S1]``, ``[S2]``, ... markers appear in the user message, or
returns exactly ``NICHT_GEFUNDEN`` when none do -- which is what the real
prompt asks the model to do with an empty or useless context. Enough to
exercise the context builder, the endpoint, and (Day 12) citation
validation without a network call or a key.

It scans the *whole* user message, so a rendered prompt whose
instructions mention ``[S1]``/``[S2]`` makes it "cite" even when the
context block is empty; pass ``canned`` to pin an exact reply for those
cases (and any other test that needs a specific answer).
"""

from __future__ import annotations

import re

from app.adapters.llm.base import LlmResponse

_MARKER = re.compile(r"\[S\d+\]")


class FakeLlmClient:
    name = "fake"

    def __init__(self, canned: str | None = None) -> None:
        self._canned = canned

    async def complete(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> LlmResponse:
        if self._canned is not None:
            return LlmResponse(text=self._canned, model=self.name)

        markers = list(dict.fromkeys(_MARKER.findall(user)))
        if not markers:
            return LlmResponse(text="NICHT_GEFUNDEN", model=self.name)

        cited = " ".join(markers)
        text = (
            f"Nach den vorliegenden Quellen {cited} gelten die dort genannten "
            f"Anforderungen. {markers[0]}"
        )
        return LlmResponse(text=text, model=self.name)
