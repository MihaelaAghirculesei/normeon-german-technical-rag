import time
from collections.abc import Callable
from typing import Any

import httpx

from app.adapters.embedding.base import Vector

#: Gemini's text-retrieval prompt formats (gemini-embedding-2 takes the task
#: in the text instead of a `taskType` field). The asymmetry plays the role
#: e5's `query: `/`passage: ` prefixes play: get it wrong and recall drops.
_QUERY_FORMAT = "task: search result | query: {}"
_DOCUMENT_FORMAT = "title: none | text: {}"

#: batchEmbedContents accepts at most 100 requests per call.
_MAX_BATCH = 100

_RETRYABLE = {429, 500, 502, 503, 504}


class GeminiEmbedder:
    """Calls the Gemini API's batchEmbedContents. The model runs at Google,
    so the backend needs no torch and fits a small instance (ADR 0005).

    `output_dimensionality` truncates the embedding to `dim` (Matryoshka;
    gemini-embedding-2 re-normalises truncated vectors), so 1024 keeps the
    schema's `vector(1024)` column. A 429 or 5xx is retried up to
    `max_retries` times, waiting `Retry-After` when the server sends it."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
        dim: int = 1024,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 30.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # The model is part of the name: embeddings rows are keyed by it, so
        # vectors from two Gemini models are never compared with each other.
        self.name = f"gemini:{model}"
        self.dim = dim
        self._model = model
        self._max_retries = max_retries
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url,
            headers={"x-goog-api-key": api_key},
            timeout=timeout,
            transport=transport,
        )

    def _post(self, texts: list[str]) -> list[Vector]:
        body = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text}]},
                    "output_dimensionality": self.dim,
                }
                for text in texts
            ]
        }
        attempt = 0
        while True:
            response = self._client.post(f"/models/{self._model}:batchEmbedContents", json=body)
            if response.status_code not in _RETRYABLE or attempt >= self._max_retries:
                break
            attempt += 1
            self._sleep(_wait_seconds(response, attempt))
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return [item["values"] for item in payload["embeddings"]]

    def _embed(self, texts: list[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for start in range(0, len(texts), _MAX_BATCH):
            vectors.extend(self._post(texts[start : start + _MAX_BATCH]))
        return vectors

    def embed_passages(self, texts: list[str]) -> list[Vector]:
        return self._embed([_DOCUMENT_FORMAT.format(t) for t in texts])

    def embed_query(self, text: str) -> Vector:
        return self._embed([_QUERY_FORMAT.format(text)])[0]


def _wait_seconds(response: httpx.Response, attempt: int) -> float:
    """`Retry-After` in seconds when present (capped at 60), otherwise
    exponential backoff from 2 s."""
    header = response.headers.get("retry-after")
    if header is not None:
        try:
            return min(float(header), 60.0)
        except ValueError:
            pass
    return float(2**attempt)
