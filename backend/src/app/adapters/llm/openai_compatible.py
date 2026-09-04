"""An ``LlmClient`` for any server that speaks the OpenAI
``/chat/completions`` wire format -- OpenAI itself, but also vLLM, TGI,
llama.cpp's server, LiteLLM, OpenRouter and most hosted gateways. One
wire format reaches every model the Week 4 matrix needs; committing to
specific provider SDKs is deferred to Day 15.

Timeouts and retries are Day 13 -- this issues a single request and lets
the error propagate. A ``transport`` can be injected for tests.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.adapters.llm.base import LlmResponse


class OpenAICompatibleClient:
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    async def complete(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> LlmResponse:
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        content = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage") or {}
        return LlmResponse(
            text=str(content).strip(),
            model=str(payload.get("model", self._model)),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
