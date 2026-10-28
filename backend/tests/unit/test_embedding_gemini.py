import json
from typing import Any

import httpx
import pytest

from app.adapters.embedding.gemini import GeminiEmbedder


def _recording_transport(
    statuses: list[int] | None = None, headers: dict[str, str] | None = None
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """Answers each batch with one 2-d vector per request; `statuses` are
    returned first, in order, before the successful answers."""
    seen: list[httpx.Request] = []
    pending = list(statuses or [])

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if pending:
            return httpx.Response(pending.pop(0), headers=headers or {}, json={"error": {}})
        body = json.loads(request.read())
        vectors = [{"values": [float(i), 0.5]} for i in range(len(body["requests"]))]
        return httpx.Response(200, json={"embeddings": vectors})

    return httpx.MockTransport(handler), seen


def _embedder(transport: httpx.MockTransport, **kwargs: Any) -> GeminiEmbedder:
    return GeminiEmbedder(api_key="secret-key", dim=2, transport=transport, **kwargs)


def _body(request: httpx.Request) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(request.read())
    return body


def test_passages_and_queries_use_the_asymmetric_retrieval_formats() -> None:
    transport, seen = _recording_transport()
    embedder = _embedder(transport)

    embedder.embed_passages(["Bremsanlage"])
    embedder.embed_query("Wie stark muss die Bremse sein?")

    passage, query = (_body(r)["requests"][0]["content"]["parts"][0]["text"] for r in seen)
    assert passage == "title: none | text: Bremsanlage"
    assert query == "task: search result | query: Wie stark muss die Bremse sein?"


def test_request_names_the_model_truncates_to_dim_and_sends_the_key() -> None:
    transport, seen = _recording_transport()
    embedder = _embedder(transport, model="gemini-embedding-2")

    vector = embedder.embed_query("Kennzeichen")

    request = seen[0]
    assert request.url.path.endswith("/models/gemini-embedding-2:batchEmbedContents")
    assert request.headers["x-goog-api-key"] == "secret-key"
    item = _body(request)["requests"][0]
    assert item["model"] == "models/gemini-embedding-2"
    assert item["output_dimensionality"] == 2
    assert vector == [0.0, 0.5]


def test_name_carries_the_model_so_two_models_never_share_rows() -> None:
    transport, _ = _recording_transport()
    assert _embedder(transport, model="gemini-embedding-2").name == "gemini:gemini-embedding-2"
    assert _embedder(transport, model="gemini-embedding-001").name == "gemini:gemini-embedding-001"


def test_more_than_a_hundred_passages_are_split_into_batches_in_order() -> None:
    transport, seen = _recording_transport()
    embedder = _embedder(transport)

    vectors = embedder.embed_passages([f"Abschnitt {i}" for i in range(250)])

    assert [len(_body(r)["requests"]) for r in seen] == [100, 100, 50]
    assert len(vectors) == 250
    assert vectors[0] == [0.0, 0.5] and vectors[100] == [0.0, 0.5] and vectors[249] == [49.0, 0.5]


def test_a_rate_limit_is_retried_after_the_servers_retry_after() -> None:
    waits: list[float] = []
    transport, seen = _recording_transport(statuses=[429], headers={"retry-after": "7"})
    embedder = _embedder(transport, sleep=waits.append)

    vector = embedder.embed_query("Scheinwerfer")

    assert len(seen) == 2
    assert waits == [7.0]
    assert vector == [0.0, 0.5]


def test_retries_are_bounded_and_then_the_error_surfaces() -> None:
    waits: list[float] = []
    transport, seen = _recording_transport(statuses=[503, 503, 503])
    embedder = _embedder(transport, max_retries=2, sleep=waits.append)

    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed_query("Scheinwerfer")

    assert len(seen) == 3
    assert waits == [2.0, 4.0]


def test_a_client_error_is_not_retried() -> None:
    waits: list[float] = []
    transport, seen = _recording_transport(statuses=[400])
    embedder = _embedder(transport, sleep=waits.append)

    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed_query("Scheinwerfer")

    assert len(seen) == 1
    assert waits == []
