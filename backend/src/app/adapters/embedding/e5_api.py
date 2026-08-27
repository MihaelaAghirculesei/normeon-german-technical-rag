from typing import Any

import httpx

from app.adapters.embedding.base import Vector, add_passage_prefix, add_query_prefix


class ApiE5Embedder:
    """Calls a remote OpenAI-compatible embeddings endpoint serving the same
    e5 model family (e.g. a self-hosted TEI server). The prefixing rule is
    identical to the local adapter -- the remote server only serves the raw
    model, so the client is still responsible for it."""

    name = "e5_api"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        dim: int,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.dim = dim
        self._model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    def _embed(self, texts: list[str]) -> list[Vector]:
        response = self._client.post(
            "/embeddings",
            json={"model": self._model, "input": texts, "dimensions": self.dim},
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return [item["embedding"] for item in payload["data"]]

    def embed_passages(self, texts: list[str]) -> list[Vector]:
        return self._embed([add_passage_prefix(t) for t in texts])

    def embed_query(self, text: str) -> Vector:
        return self._embed([add_query_prefix(text)])[0]
