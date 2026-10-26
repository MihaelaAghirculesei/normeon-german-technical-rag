"""Per-IP rate limits on the endpoints that cost something (Giorno 29).

A public demo link is an open door: every question spends free-tier
Gemini quota, and the login endpoint invites password guessing. Each
rule allows `limit` requests per client IP in a sliding `window_s`;
beyond that the client gets 429 with `Retry-After`, before the app runs.

The counters live in this process. The demo runs one instance, so that is
the whole truth; several instances would each allow the limit (a shared
store such as Redis would be needed then). The client IP is the ASGI
scope's, which uvicorn's `--proxy-headers` sets from the platform's
`X-Forwarded-For`.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from app.core.asgi import ASGIApp, Receive, Scope, Send

#: Past this many tracked (rule, IP) keys, clients whose hits have all left
#: their window are forgotten, so memory follows the active clients, not
#: every address ever seen.
_SWEEP_AT = 10_000


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    method: str
    paths: frozenset[str]
    limit: int
    window_s: float


class RateLimit:
    def __init__(
        self,
        app: ASGIApp,
        *,
        rules: list[Rule],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app = app
        self.rules = [rule for rule in rules if rule.limit > 0]
        self._clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = {}

    def _rule_for(self, scope: Scope) -> Rule | None:
        for rule in self.rules:
            if scope["method"] == rule.method and scope["path"] in rule.paths:
                return rule
        return None

    def _retry_after(self, rule: Rule, ip: str) -> float | None:
        """Seconds until the client may retry, or None if this request is
        allowed (and then counted)."""
        now = self._clock()
        if len(self._hits) > _SWEEP_AT:
            self._sweep(now)
        hits = self._hits.setdefault((rule.name, ip), deque())
        while hits and hits[0] <= now - rule.window_s:
            hits.popleft()
        if len(hits) >= rule.limit:
            return hits[0] + rule.window_s - now
        hits.append(now)
        return None

    def _sweep(self, now: float) -> None:
        windows = {rule.name: rule.window_s for rule in self.rules}
        self._hits = {
            key: hits
            for key, hits in self._hits.items()
            if hits and hits[-1] > now - windows[key[0]]
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        rule = self._rule_for(scope) if scope["type"] == "http" else None
        if rule is None:
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        ip = client[0] if client else "unknown"
        wait = self._retry_after(rule, ip)
        if wait is None:
            await self.app(scope, receive, send)
            return

        body = json.dumps(
            {"code": "rate_limited", "detail": f"too many requests ({rule.name}), retry later"}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"retry-after", str(max(1, math.ceil(wait))).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
