"""One request id per request, everywhere (Giorno 27).

The middleware takes the caller's `X-Request-ID` when it looks like an id
(so a proxy's or the frontend's id carries through), or makes one. It
binds the id to structlog's context variables, so every log line written
while handling the request carries it; returns it as `X-Request-ID`; and
logs one `request.completed` line with method, path, status and duration.
`query_logs.request_id` uses the same id, so an answer in the database,
its log lines and its trace can be matched up.

Background work started by a request (document ingestion) runs inside the
same request call, so it logs under the same id.
"""

from __future__ import annotations

import re
import time
import uuid
from contextvars import ContextVar

import structlog

from app.core.asgi import ASGIApp, Message, Receive, Scope, Send
from app.core.observability import REQUEST_ID_ATTRIBUTE, tag_current_span

_HEADER = b"x-request-id"
_ACCEPTABLE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_QUIET_PATHS = frozenset({"/health", "/metrics"})

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_log = structlog.get_logger(__name__)


def current_request_id() -> str | None:
    """The id of the request being handled, or None outside a request."""
    return _request_id.get()


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        offered = dict(scope["headers"]).get(_HEADER, b"").decode("latin-1")
        request_id = offered if _ACCEPTABLE.match(offered) else uuid.uuid4().hex
        token = _request_id.set(request_id)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        # The request's trace span (when tracing wraps this middleware).
        tag_current_span(**{REQUEST_ID_ATTRIBUTE: request_id})
        started = time.perf_counter()
        status = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [*message.get("headers", []), (_HEADER, request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            if scope["path"] not in _QUIET_PATHS:
                _log.info(
                    "request.completed",
                    method=scope["method"],
                    path=scope["path"],
                    status=status,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
            structlog.contextvars.unbind_contextvars("request_id")
            _request_id.reset(token)
