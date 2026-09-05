"""Endpoint-level tests for POST /api/v1/chat.

The DB session, embedder, reranker, LLM client and ``generate_answer``
itself are all stubbed -- these check the HTTP contract (validation,
response shape, parameter forwarding, the backend-held page mapping)
without a database or a model.
"""

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_embedder, get_llm_client, get_reranker, get_session
from app.api.v1 import chat
from app.core.errors import LlmUnavailableError
from app.domain.citations import Citation
from app.domain.context import Source
from app.domain.models import PipelineTiming
from app.main import app
from app.services.generation import AnswerResult, StreamEvent

TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _source(marker: str, page: int) -> Source:
    return Source(
        marker=marker,
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="StVZO.pdf",
        page_from=page,
        page_to=page,
        section_path="50",
        heading="Scheinwerfer",
        content="…",
    )


def _citation(marker: str, page: int) -> Citation:
    return Citation(
        marker=marker,
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="StVZO.pdf",
        page_from=page,
        page_to=page,
        section_path="50",
        snippet="Scheinwerfer muessen ...",
    )


def _answer(
    text: str = "Laut [S1] gilt die Regel. [S1]",
    citations: list[Citation] | None = None,
    cost_usd: float | None = None,
) -> AnswerResult:
    return AnswerResult(
        answer=text,
        sources=[_source("S1", 14), _source("S2", 15)],
        citations=citations if citations is not None else [_citation("S1", 14)],
        prompt_name="answer_de.v1",
        prompt_sha256="a" * 64,
        model="fake",
        retrieval_timing=PipelineTiming(
            hybrid_ms=10.0, rerank_ms=5.0, select_ms=1.0, total_ms=16.0
        ),
        generation_ms=3.2,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=cost_usd,
    )


@pytest.fixture
def client_with_generator(monkeypatch: pytest.MonkeyPatch) -> Any:
    def make(result: AnswerResult) -> tuple[TestClient, list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []

        async def fake_generate_answer(
            session: Any,
            embedder: Any,
            reranker: Any,
            llm: Any,
            *,
            tenant_id: uuid.UUID,
            question: str,
            strategy: str | None = None,
        ) -> AnswerResult:
            calls.append(
                {"tenant_id": tenant_id, "question": question, "strategy": strategy}
            )
            return result

        monkeypatch.setattr(chat, "generate_answer", fake_generate_answer)
        app.dependency_overrides[get_session] = lambda: object()
        app.dependency_overrides[get_embedder] = lambda: object()
        app.dependency_overrides[get_reranker] = lambda: object()
        app.dependency_overrides[get_llm_client] = lambda: object()
        return TestClient(app), calls

    yield make
    app.dependency_overrides.clear()


def test_chat_returns_answer_sources_and_prompt_identity(client_with_generator: Any) -> None:
    client, _ = client_with_generator(_answer())

    response = client.post(
        "/api/v1/chat",
        json={"question": "Was gilt fuer Scheinwerfer?", "tenant_id": str(TENANT_ID)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Laut [S1] gilt die Regel. [S1]"
    assert [s["marker"] for s in body["sources"]] == ["S1", "S2"]
    assert body["prompt_name"] == "answer_de.v1"
    assert len(body["prompt_sha256"]) == 64
    assert body["model"] == "fake"
    assert body["retrieval_timing"]["total_ms"] == 16.0
    assert body["generation_ms"] == 3.2
    assert body["cost_usd"] is None


def test_chat_response_carries_a_priced_cost(client_with_generator: Any) -> None:
    client, _ = client_with_generator(_answer(cost_usd=0.00045))

    body = client.post(
        "/api/v1/chat", json={"question": "x", "tenant_id": str(TENANT_ID)}
    ).json()

    assert body["cost_usd"] == 0.00045


def test_chat_response_carries_the_backend_held_page_mapping(
    client_with_generator: Any,
) -> None:
    client, _ = client_with_generator(_answer())

    body = client.post(
        "/api/v1/chat",
        json={"question": "x", "tenant_id": str(TENANT_ID)},
    ).json()

    s1 = body["sources"][0]
    assert s1["marker"] == "S1"
    assert s1["page_from"] == 14 and s1["page_to"] == 14
    assert s1["filename"] == "StVZO.pdf"
    assert s1["section_path"] == "50"
    # the model's text never contains a page number; the mapping is here
    assert "14" not in body["answer"]


def test_chat_forwards_question_tenant_and_strategy(client_with_generator: Any) -> None:
    client, calls = client_with_generator(_answer())

    client.post(
        "/api/v1/chat",
        json={
            "question": "Welche Lenkkraft nennt LH-3.2.1?",
            "tenant_id": str(TENANT_ID),
            "strategy": "structural",
        },
    )

    assert calls == [
        {
            "tenant_id": TENANT_ID,
            "question": "Welche Lenkkraft nennt LH-3.2.1?",
            "strategy": "structural",
        }
    ]


def test_chat_response_carries_validated_citations(client_with_generator: Any) -> None:
    client, _ = client_with_generator(_answer(citations=[_citation("S1", 14)]))

    body = client.post(
        "/api/v1/chat", json={"question": "x", "tenant_id": str(TENANT_ID)}
    ).json()

    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["marker"] == "S1"
    assert citation["page_from"] == 14 and citation["page_to"] == 14
    assert citation["filename"] == "StVZO.pdf"
    assert citation["section_path"] == "50"
    assert citation["snippet"] == "Scheinwerfer muessen ..."
    assert "chunk_id" in citation and "document_id" in citation


def test_chat_passes_nicht_gefunden_through_untouched(client_with_generator: Any) -> None:
    client, _ = client_with_generator(
        AnswerResult(
            answer="NICHT_GEFUNDEN",
            sources=[],
            citations=[],
            prompt_name="answer_de.v1",
            prompt_sha256="b" * 64,
            model="fake",
            retrieval_timing=PipelineTiming(0.0, 0.0, 0.0, 0.0),
            generation_ms=0.1,
            prompt_tokens=None,
            completion_tokens=None,
            cost_usd=None,
        )
    )

    body = client.post(
        "/api/v1/chat", json={"question": "Oelpreis?", "tenant_id": str(TENANT_ID)}
    ).json()

    assert body["answer"] == "NICHT_GEFUNDEN"
    assert body["sources"] == []
    assert body["citations"] == []


def test_chat_rejects_a_blank_question(client_with_generator: Any) -> None:
    client, _ = client_with_generator(_answer())

    response = client.post(
        "/api/v1/chat", json={"question": "", "tenant_id": str(TENANT_ID)}
    )

    assert response.status_code == 422


def test_chat_rejects_a_bad_tenant_id(client_with_generator: Any) -> None:
    client, _ = client_with_generator(_answer())

    response = client.post(
        "/api/v1/chat", json={"question": "x", "tenant_id": "not-a-uuid"}
    )

    assert response.status_code == 422


def test_chat_rejects_an_unknown_strategy(client_with_generator: Any) -> None:
    client, _ = client_with_generator(_answer())

    response = client.post(
        "/api/v1/chat",
        json={"question": "x", "tenant_id": str(TENANT_ID), "strategy": "semantic"},
    )

    assert response.status_code == 422


def test_an_llm_failure_never_surfaces_as_a_bare_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Giorno 13: a typed NormeonError from the generation path is mapped
    to a coherent JSON error body by main.py's generic handler, not left
    to fall through as an unstructured crash."""

    async def raising_generate_answer(*args: Any, **kwargs: Any) -> AnswerResult:
        raise LlmUnavailableError("the model host is down")

    monkeypatch.setattr(chat, "generate_answer", raising_generate_answer)
    app.dependency_overrides[get_session] = lambda: object()
    app.dependency_overrides[get_embedder] = lambda: object()
    app.dependency_overrides[get_reranker] = lambda: object()
    app.dependency_overrides[get_llm_client] = lambda: object()
    try:
        response = TestClient(app).post(
            "/api/v1/chat", json={"question": "x", "tenant_id": str(TENANT_ID)}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {"code": "llm_unavailable", "detail": "the model host is down"}


# --- POST /api/v1/chat/stream (Giorno 14) -----------------------------------


def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip("\n").split("\n\n"):
        if not block or block.startswith(":"):
            continue
        event_line, data_line = block.split("\n", 1)
        events.append(
            (event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: ")))
        )
    return events


def _stream_with(events: list[StreamEvent]) -> Any:
    async def fake_generate_answer_stream(
        session: Any,
        embedder: Any,
        reranker: Any,
        llm: Any,
        *,
        tenant_id: uuid.UUID,
        question: str,
        strategy: str | None = None,
    ) -> AsyncGenerator[StreamEvent]:
        for event in events:
            yield event

    return fake_generate_answer_stream


@pytest.fixture
def stream_client(monkeypatch: pytest.MonkeyPatch) -> Any:
    def make(events: list[StreamEvent]) -> TestClient:
        monkeypatch.setattr(chat, "generate_answer_stream", _stream_with(events))
        app.dependency_overrides[get_session] = lambda: object()
        app.dependency_overrides[get_embedder] = lambda: object()
        app.dependency_overrides[get_reranker] = lambda: object()
        app.dependency_overrides[get_llm_client] = lambda: object()
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def test_chat_stream_emits_sse_events_in_order(stream_client: Any) -> None:
    client = stream_client(
        [
            StreamEvent("stage", {"stage": "retrieving"}),
            StreamEvent("stage", {"stage": "reranking"}),
            StreamEvent("sources", {"sources": [_source("S1", 14)]}),
            StreamEvent("stage", {"stage": "generating"}),
            StreamEvent("token", {"text": "Die "}),
            StreamEvent("token", {"text": "Antwort."}),
            StreamEvent(
                "done",
                {
                    "latency_ms": {"retrieval_ms": 10.0, "generation_ms": 5.0, "total_ms": 15.0},
                    "cost_usd": None,
                    "citations": [_citation("S1", 14)],
                },
            ),
        ]
    )

    with client.stream(
        "POST", "/api/v1/chat/stream", json={"question": "x", "tenant_id": str(TENANT_ID)}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    events = _parse_sse(body)
    assert [e for e, _ in events] == [
        "stage", "stage", "sources", "stage", "token", "token", "done",
    ]
    assert events[0][1] == {"stage": "retrieving"}
    assert events[4][1] == {"text": "Die "}

    sources_data = events[2][1]["sources"]
    assert [s["marker"] for s in sources_data] == ["S1"]
    assert sources_data[0]["page_from"] == 14

    done_data = events[-1][1]
    assert done_data["cost_usd"] is None
    assert [c["marker"] for c in done_data["citations"]] == ["S1"]
    assert done_data["latency_ms"]["total_ms"] == 15.0


def test_chat_stream_passes_an_error_event_through(stream_client: Any) -> None:
    client = stream_client(
        [
            StreamEvent("stage", {"stage": "retrieving"}),
            StreamEvent("error", {"code": "llm_timeout", "message": "boom"}),
        ]
    )

    with client.stream(
        "POST", "/api/v1/chat/stream", json={"question": "x", "tenant_id": str(TENANT_ID)}
    ) as response:
        body = "".join(response.iter_text())

    events = _parse_sse(body)
    assert events[-1] == ("error", {"code": "llm_timeout", "message": "boom"})


def test_chat_stream_rejects_a_blank_question_before_streaming_anything(
    stream_client: Any,
) -> None:
    client = stream_client([StreamEvent("stage", {"stage": "retrieving"})])

    response = client.post(
        "/api/v1/chat/stream", json={"question": "", "tenant_id": str(TENANT_ID)}
    )

    assert response.status_code == 422
