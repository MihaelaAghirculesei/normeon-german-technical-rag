"""An ``LlmClient`` for any server that speaks the OpenAI
``/chat/completions`` wire format -- OpenAI itself, but also vLLM, TGI,
llama.cpp's server, LiteLLM, OpenRouter and most hosted gateways. One
wire format reaches every model the Week 4 matrix needs; committing to
specific provider SDKs is deferred to Day 15.

Transient failures (a timeout, a dropped connection, a 5xx) are retried
with `tenacity`, up to `max_retries` extra attempts with a short
exponential backoff -- a per-request budget, not unbounded retrying. A
4xx is never retried (retrying a bad request or an auth failure just
wastes the budget). Whatever finally fails is re-raised as one of the
typed `core.errors.NormeonError`s, never a raw `httpx` exception, so the
API's generic error handler always returns a coherent JSON body instead
of a bare 500. A ``transport`` can be injected for tests.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.adapters.llm.base import LlmResponse
from app.core.errors import LlmTimeoutError, LlmUnavailableError


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError))


class OpenAICompatibleClient:
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )
        self._max_retries = max_retries

    async def complete(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> LlmResponse:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = await self._post_with_retry(body)
        except httpx.TimeoutException as exc:
            raise LlmTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LlmUnavailableError(str(exc)) from exc

        payload: dict[str, Any] = response.json()
        content = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage") or {}
        return LlmResponse(
            text=str(content).strip(),
            model=str(payload.get("model", self._model)),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def _post_with_retry(self, body: dict[str, Any]) -> httpx.Response:
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.5, max=4),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                response = await self._client.post("/chat/completions", json=body)
                response.raise_for_status()
                return response
        raise AssertionError("unreachable: tenacity always raises or returns above")
