"""Refuse an oversized request body before reading it (Giorno 26).

By the time an endpoint runs, Starlette has already received the whole
multipart body: a check there would only fire after a 200 MB upload had
been transferred and spooled to disk. This middleware decides from the
headers instead, before the app sees the request:
  - a declared `Content-Length` over the limit gets 413 at once;
  - a body without a declared length (chunked transfer) gets 411, so the
    limit cannot be dodged by not announcing the size. Browsers always
    send `Content-Length` for a form upload, so this costs real clients
    nothing.
The server (h11) does not accept more bytes than were declared, so the
declared length is a real bound.

Document uploads get `max_upload_bytes`; every other endpoint takes small
JSON and gets `max_request_bytes`.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_UPLOAD_PATH = "/api/v1/documents"
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})


class BodySizeLimit:
    def __init__(self, app: ASGIApp, *, max_upload_bytes: int, max_request_bytes: int) -> None:
        self.app = app
        self.max_upload_bytes = max_upload_bytes
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope["headers"]}
        limit = (
            self.max_upload_bytes
            if scope["path"].rstrip("/") == _UPLOAD_PATH
            else self.max_request_bytes
        )
        declared = headers.get(b"content-length")
        if declared is None:
            if b"chunked" in headers.get(b"transfer-encoding", b"").lower():
                await _refuse(send, 411, "length_required", "send a Content-Length header")
                return
        elif not declared.isdigit():
            await _refuse(send, 400, "bad_request", "invalid Content-Length header")
            return
        elif int(declared) > limit:
            await _refuse(send, 413, "upload_too_large", f"at most {_size(limit)} are accepted")
            return
        await self.app(scope, receive, send)


def _size(n: int) -> str:
    return f"{n // 2**20} MB" if n >= 2**20 else f"{n // 1024} KB"


async def _refuse(send: Send, status: int, code: str, detail: str) -> None:
    body = json.dumps({"code": code, "detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                # The client is mid-upload; don't keep the connection.
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
