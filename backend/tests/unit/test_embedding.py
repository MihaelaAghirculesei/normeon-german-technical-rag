import json
import threading
import time
from typing import Any

import httpx
import pytest

from app.adapters.embedding.base import add_passage_prefix, add_query_prefix
from app.adapters.embedding.e5_api import ApiE5Embedder
from app.adapters.embedding.e5_local import LocalE5Embedder


def test_passage_prefix() -> None:
    assert add_passage_prefix("Fahrzeugzulassung") == "passage: Fahrzeugzulassung"


def test_query_prefix() -> None:
    assert add_query_prefix("Was ist LH-3.2.1?") == "query: Was ist LH-3.2.1?"


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeSentenceTransformer:
    def __init__(self) -> None:
        self.seen_inputs: list[Any] = []

    def encode(self, inputs: Any, normalize_embeddings: bool = True) -> Any:
        self.seen_inputs.append(inputs)
        if isinstance(inputs, list):
            return [_FakeVector([1.0, 2.0]) for _ in inputs]
        return _FakeVector([3.0, 4.0])


def test_local_embedder_prefixes_passages_before_encoding() -> None:
    embedder = LocalE5Embedder(model_name="unused", dim=2)
    fake = _FakeSentenceTransformer()
    embedder._model = fake  # type: ignore[assignment]

    vectors = embedder.embed_passages(["Fahrzeugzulassung", "Betriebserlaubnis"])

    assert fake.seen_inputs == [["passage: Fahrzeugzulassung", "passage: Betriebserlaubnis"]]
    assert vectors == [[1.0, 2.0], [1.0, 2.0]]


def test_local_embedder_prefixes_query_before_encoding() -> None:
    embedder = LocalE5Embedder(model_name="unused", dim=2)
    fake = _FakeSentenceTransformer()
    embedder._model = fake  # type: ignore[assignment]

    vector = embedder.embed_query("Was ist LH-3.2.1?")

    assert fake.seen_inputs == ["query: Was ist LH-3.2.1?"]
    assert vector == [3.0, 4.0]


def test_local_embedder_does_not_load_model_until_first_use() -> None:
    embedder = LocalE5Embedder(model_name="unused", dim=2)
    assert embedder._model is None


def test_local_embedder_loads_the_model_only_once_under_concurrent_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: the eval runner (Giorno 17-18) drives several
    questions concurrently, each calling `embed_query`/`embed_passages`
    off-thread. An unlocked lazy-load let multiple threads race to
    construct a SentenceTransformer at once, surfacing as PyTorch's
    "Cannot copy out of meta tensor; no data!" -- reproduced running the
    real 50-question eval set. This proves the fix: exactly one
    construction no matter how many threads arrive concurrently."""
    construct_count = 0
    count_lock = threading.Lock()

    class _SlowSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            nonlocal construct_count
            with count_lock:
                construct_count += 1
            time.sleep(0.05)  # widen the race window past the naive check

    monkeypatch.setattr(
        "app.adapters.embedding.e5_local.SentenceTransformer", _SlowSentenceTransformer
    )
    embedder = LocalE5Embedder(model_name="unused", dim=2)

    threads = [threading.Thread(target=lambda: embedder._loaded_model) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert construct_count == 1


def _recording_transport() -> tuple[httpx.MockTransport, list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        payloads.append(payload)
        embeddings = [{"embedding": [0.1, 0.2]} for _ in payload["input"]]
        return httpx.Response(200, json={"data": embeddings})

    return httpx.MockTransport(handler), payloads


def test_api_embedder_prefixes_and_parses_passages() -> None:
    transport, payloads = _recording_transport()
    embedder = ApiE5Embedder(
        base_url="https://example.test", api_key=None, model="e5", dim=2, transport=transport
    )

    vectors = embedder.embed_passages(["Fahrzeugzulassung"])

    assert payloads[-1]["input"] == ["passage: Fahrzeugzulassung"]
    assert vectors == [[0.1, 0.2]]


def test_api_embedder_prefixes_and_parses_query() -> None:
    transport, payloads = _recording_transport()
    embedder = ApiE5Embedder(
        base_url="https://example.test", api_key=None, model="e5", dim=2, transport=transport
    )

    vector = embedder.embed_query("Was ist LH-3.2.1?")

    assert payloads[-1]["input"] == ["query: Was ist LH-3.2.1?"]
    assert vector == [0.1, 0.2]


def test_api_embedder_sends_bearer_token_when_api_key_set() -> None:
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

    embedder = ApiE5Embedder(
        base_url="https://example.test",
        api_key="secret-key",
        model="e5",
        dim=2,
        transport=httpx.MockTransport(handler),
    )

    embedder.embed_query("hello")

    assert seen_headers["authorization"] == "Bearer secret-key"
