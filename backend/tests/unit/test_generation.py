"""Unit tests for ``generate_answer`` composition. ``retrieve_context`` is
stubbed on the ``generation`` module, so no database or real models are
touched; the LLM is the FakeLlmClient (or a spy).
"""

import uuid
from collections.abc import AsyncGenerator
from typing import Any

from app.adapters.llm.base import Delta, LlmResponse
from app.adapters.llm.fake import FakeLlmClient
from app.core.errors import LlmTimeoutError
from app.core.metrics import hallucinated_citation_total
from app.domain.models import PipelineTiming, RetrievalResult, RetrievedChunk
from app.services import generation

TENANT = uuid.uuid4()


def _chunk(
    section: str,
    *,
    score: float = 0.8,
    version_label: str | None = None,
    content: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="StVZO.pdf",
        content=content if content is not None else f"Regelungstext zu Abschnitt {section}.",
        page_from=12,
        page_to=12,
        section_path=section,
        heading=None,
        score=score,
        version_label=version_label,
    )


def _stub_retrieval(monkeypatch: Any, chunks: list[RetrievedChunk]) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def fake_retrieve_context(
        session: Any,
        embedder: Any,
        reranker: Any,
        *,
        tenant_id: uuid.UUID,
        question: str,
        strategy: str | None = None,
    ) -> RetrievalResult:
        seen["tenant_id"] = tenant_id
        seen["question"] = question
        seen["strategy"] = strategy
        timing = PipelineTiming(
            hybrid_ms=10.0, rerank_ms=5.0, select_ms=1.0, total_ms=16.0
        )
        return RetrievalResult(context=chunks, reranked=chunks, timing=timing)

    monkeypatch.setattr(generation, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(generation, "record_query_log", _noop_record_query_log)
    return seen


async def _noop_record_query_log(session: Any, **kwargs: Any) -> None:
    """Default query_log stub (Giorno 15): most tests don't care that a
    row would be written, just that generate_answer/_stream don't need a
    real session to do it -- `object()` is still a valid `session` arg."""


def _stub_query_log(monkeypatch: Any) -> list[dict[str, Any]]:
    """Overrides the no-op stub above for tests that need to inspect
    what record_query_log was called with."""
    calls: list[dict[str, Any]] = []

    async def fake_record_query_log(session: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(generation, "record_query_log", fake_record_query_log)
    return calls


async def test_composes_retrieval_context_prompt_and_llm(monkeypatch: Any) -> None:
    seen = _stub_retrieval(monkeypatch, [_chunk("5.1"), _chunk("5.2")])

    result = await generation.generate_answer(
        object(), object(), object(), FakeLlmClient(),
        tenant_id=TENANT, question="Welche Lenkkraft?", strategy="structural",
    )

    assert seen == {
        "tenant_id": TENANT,
        "question": "Welche Lenkkraft?",
        "strategy": "structural",
    }
    assert [s.marker for s in result.sources] == ["S1", "S2"]
    assert [c.marker for c in result.citations] == ["S1", "S2"]
    assert result.prompt_name == "answer_de.v1"
    assert len(result.prompt_sha256) == 64
    assert result.model == "fake"
    assert "[S1]" in result.answer
    assert result.retrieval_timing.total_ms == 16.0
    assert result.generation_ms >= 0


async def test_the_model_sees_the_rendered_prompt_with_the_context_block(
    monkeypatch: Any,
) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1")])
    captured: dict[str, str] = {}

    class _Spy:
        name = "spy"

        async def complete(
            self, *, system: str, user: str, temperature: float, max_tokens: int
        ) -> LlmResponse:
            captured["system"] = system
            captured["user"] = user
            return LlmResponse(text="ok [S1]", model="spy")

    await generation.generate_answer(
        object(), object(), object(), _Spy(),
        tenant_id=TENANT, question="Frage X?",
    )

    assert "[S1] Quelle: StVZO.pdf | Seite 12 | Abschnitt 5.1" in captured["user"]
    assert "Frage: Frage X?" in captured["user"]
    assert "Antworte auf Deutsch." in captured["user"]


class _RefusingLlm:
    """An LLM double that fails the test if it is ever called -- used to
    prove the pre-generation abstention gate genuinely skips generation."""

    name = "refusing"

    async def complete(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> LlmResponse:
        raise AssertionError("the LLM should not have been called")


async def test_empty_retrieval_abstains_pre_generation_without_calling_the_llm(
    monkeypatch: Any,
) -> None:
    _stub_retrieval(monkeypatch, [])

    result = await generation.generate_answer(
        object(), object(), object(), _RefusingLlm(),
        tenant_id=TENANT, question="Wie hoch ist der Oelpreis?",
    )

    assert result.sources == []
    assert result.citations == []
    assert result.answer == "NICHT_GEFUNDEN"
    assert result.generation_ms == 0.0


async def test_a_low_rerank_score_abstains_pre_generation_without_calling_the_llm(
    monkeypatch: Any,
) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1", score=-3.0)])

    result = await generation.generate_answer(
        object(), object(), object(), _RefusingLlm(),
        tenant_id=TENANT, question="Wie hoch ist der Oelpreis in Katar?",
    )

    assert result.answer == "NICHT_GEFUNDEN"
    assert result.sources == []
    assert result.model == "refusing"


async def test_a_rerank_score_at_or_above_the_threshold_still_calls_the_llm(
    monkeypatch: Any,
) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1", score=0.0)])

    result = await generation.generate_answer(
        object(), object(), object(), FakeLlmClient(),
        tenant_id=TENANT, question="Frage?",
    )

    assert result.answer != "NICHT_GEFUNDEN"
    assert result.sources != []


async def test_an_invented_citation_marker_is_dropped_and_counted(
    monkeypatch: Any,
) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1")])

    class _Spy:
        name = "spy"

        async def complete(
            self, *, system: str, user: str, temperature: float, max_tokens: int
        ) -> LlmResponse:
            return LlmResponse(text="Es gilt Regel A [S1] und angeblich [S9].", model="spy")

    before = hallucinated_citation_total.value

    result = await generation.generate_answer(
        object(), object(), object(), _Spy(),
        tenant_id=TENANT, question="Frage?",
    )

    assert "[S9]" not in result.answer
    assert "[S1]" in result.answer
    assert [c.marker for c in result.citations] == ["S1"]
    assert hallucinated_citation_total.value == before + 1


async def test_claims_with_zero_valid_citations_become_an_abstention(
    monkeypatch: Any,
) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1")])

    result = await generation.generate_answer(
        object(), object(), object(),
        FakeLlmClient(canned="Angeblich gilt Regel Z [S9]."),
        tenant_id=TENANT, question="Frage?",
    )

    assert result.answer == "NICHT_GEFUNDEN"
    assert result.citations == []


async def test_prompt_name_override_is_honoured(monkeypatch: Any) -> None:
    _stub_retrieval(monkeypatch, [_chunk("1")])

    result = await generation.generate_answer(
        object(), object(), object(), FakeLlmClient(),
        tenant_id=TENANT, question="q", prompt_name="answer_de.v1",
    )

    assert result.prompt_name == "answer_de.v1"


async def test_conflicting_requirement_code_versions_add_an_instruction_to_the_system(
    monkeypatch: Any,
) -> None:
    """The corpus's synthetic Lastenheft-EPS v1.2 / v2.0 scenario: two
    different documents, different version_labels, restating the same
    requirement code with different values."""
    _stub_retrieval(
        monkeypatch,
        [
            _chunk("3.2.1", content="LH-3.2.1 fordert 300 N.", version_label="v1.2"),
            _chunk("3.2.1", content="LH-3.2.1 fordert 250 N.", version_label="v2.0"),
        ],
    )
    captured: dict[str, str] = {}

    class _Spy:
        name = "spy"

        async def complete(
            self, *, system: str, user: str, temperature: float, max_tokens: int
        ) -> LlmResponse:
            captured["system"] = system
            return LlmResponse(text="[S1]", model="spy")

    await generation.generate_answer(
        object(), object(), object(), _Spy(), tenant_id=TENANT, question="Frage?",
    )

    assert "unterschiedlichen Versionen" in captured["system"]


async def test_a_single_version_label_adds_no_conflict_instruction(
    monkeypatch: Any,
) -> None:
    _stub_retrieval(
        monkeypatch,
        [
            _chunk("3.2.1", content="LH-3.2.1 fordert 300 N.", version_label="v1.2"),
            _chunk("5.2", content="Belangloser Text.", version_label="v1.2"),
        ],
    )
    captured: dict[str, str] = {}

    class _Spy:
        name = "spy"

        async def complete(
            self, *, system: str, user: str, temperature: float, max_tokens: int
        ) -> LlmResponse:
            captured["system"] = system
            return LlmResponse(text="[S1]", model="spy")

    await generation.generate_answer(
        object(), object(), object(), _Spy(), tenant_id=TENANT, question="Frage?",
    )

    assert "unterschiedlichen Versionen" not in captured["system"]


# --- generate_answer_stream (Giorno 14) -------------------------------------


class _RefusingStreamLlm:
    """Same idea as _RefusingLlm, for the streaming path: proves the
    pre-generation gate never even calls `stream`."""

    name = "refusing"

    def stream(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[Delta]:
        raise AssertionError("the LLM should not have been called")


class _StreamSpy:
    """Yields the given chunks and records the system/user it was called
    with; a stand-in for a real streaming LlmClient."""

    name = "spy"

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.captured: dict[str, str] = {}

    async def stream(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[Delta]:
        self.captured["system"] = system
        self.captured["user"] = user
        for chunk in self._chunks:
            yield Delta(text=chunk, model="spy")


class _FailingStreamLlm:
    name = "spy"

    async def stream(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[Delta]:
        yield Delta(text="partial ", model="spy")
        raise LlmTimeoutError("boom")


class _FinallyTrackingLlm:
    """Its `stream` sets `closed = True` in a `finally`, so closing the
    generator early (a client disconnect) can be proven to cascade all
    the way down to it -- "annulla il task LLM" from the plan."""

    name = "spy"

    def __init__(self) -> None:
        self.closed = False

    async def stream(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[Delta]:
        try:
            yield Delta(text="a", model="spy")
            yield Delta(text="b", model="spy")
        finally:
            self.closed = True


async def test_stream_emits_events_in_the_documented_order(monkeypatch: Any) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1"), _chunk("5.2")])

    events = [
        event
        async for event in generation.generate_answer_stream(
            object(), object(), object(), FakeLlmClient(),
            tenant_id=TENANT, question="Welche Lenkkraft?", strategy="structural",
        )
    ]

    kinds = [e.event for e in events]
    assert kinds[:4] == ["stage", "stage", "sources", "stage"]
    assert kinds[-1] == "done"
    assert all(k == "token" for k in kinds[4:-1])
    assert kinds.count("token") >= 1

    assert events[0].data == {"stage": "retrieving"}
    assert events[1].data == {"stage": "reranking"}
    assert [s.marker for s in events[2].data["sources"]] == ["S1", "S2"]
    assert events[3].data == {"stage": "generating"}

    full_text = "".join(e.data["text"] for e in events if e.event == "token")
    assert "[S1]" in full_text

    done = events[-1]
    assert set(done.data.keys()) == {"latency_ms", "cost_usd", "citations"}
    assert done.data["cost_usd"] is None
    assert [c.marker for c in done.data["citations"]] == ["S1", "S2"]
    assert done.data["latency_ms"]["retrieval_ms"] == 16.0
    assert done.data["latency_ms"]["generation_ms"] >= 0


async def test_stream_abstains_pre_generation_without_calling_the_llm(monkeypatch: Any) -> None:
    _stub_retrieval(monkeypatch, [])

    events = [
        event
        async for event in generation.generate_answer_stream(
            object(), object(), object(), _RefusingStreamLlm(),
            tenant_id=TENANT, question="Wie hoch ist der Oelpreis?",
        )
    ]

    assert [e.event for e in events] == ["stage", "stage", "sources", "token", "done"]
    assert events[2].data == {"sources": []}
    assert events[3].data == {"text": "NICHT_GEFUNDEN"}
    assert events[4].data["citations"] == []


async def test_stream_tokens_are_raw_but_done_citations_exclude_invented_markers(
    monkeypatch: Any,
) -> None:
    """The tension documented in generate_answer_stream's docstring: a
    marker already streamed cannot be un-sent, but `done.citations` stays
    the honest, validated list."""
    _stub_retrieval(monkeypatch, [_chunk("5.1")])
    llm = _StreamSpy(["Es gilt Regel A [S1] ", "und angeblich [S9]."])

    events = [
        event
        async for event in generation.generate_answer_stream(
            object(), object(), object(), llm, tenant_id=TENANT, question="Frage?",
        )
    ]

    streamed_text = "".join(e.data["text"] for e in events if e.event == "token")
    assert "[S9]" in streamed_text

    done = events[-1]
    assert [c.marker for c in done.data["citations"]] == ["S1"]


async def test_stream_adds_the_conflict_instruction_to_the_system_message(
    monkeypatch: Any,
) -> None:
    _stub_retrieval(
        monkeypatch,
        [
            _chunk("3.2.1", content="LH-3.2.1 fordert 300 N.", version_label="v1.2"),
            _chunk("3.2.1", content="LH-3.2.1 fordert 250 N.", version_label="v2.0"),
        ],
    )
    llm = _StreamSpy(["[S1]"])

    async for _ in generation.generate_answer_stream(
        object(), object(), object(), llm, tenant_id=TENANT, question="Frage?",
    ):
        pass

    assert "unterschiedlichen Versionen" in llm.captured["system"]


async def test_an_llm_error_mid_stream_becomes_an_error_event_not_a_dead_stream(
    monkeypatch: Any,
) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1")])

    events = [
        event
        async for event in generation.generate_answer_stream(
            object(), object(), object(), _FailingStreamLlm(),
            tenant_id=TENANT, question="Frage?",
        )
    ]

    assert events[-1].event == "error"
    assert events[-1].data == {"code": "llm_timeout", "message": "boom"}
    assert not any(e.event == "done" for e in events)
    assert any(e.event == "token" and e.data["text"] == "partial " for e in events)


async def test_closing_the_stream_early_cascades_to_the_llm_client(monkeypatch: Any) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1")])
    llm = _FinallyTrackingLlm()

    agen = generation.generate_answer_stream(
        object(), object(), object(), llm, tenant_id=TENANT, question="Frage?",
    )
    async for event in agen:
        if event.event == "token":
            break
    await agen.aclose()

    assert llm.closed is True


# --- cost tracking + query_logs (Giorno 15) ---------------------------------


async def test_generate_answer_writes_a_query_log_row(monkeypatch: Any) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1"), _chunk("5.2")])
    calls = _stub_query_log(monkeypatch)

    result = await generation.generate_answer(
        object(), object(), object(), FakeLlmClient(),
        tenant_id=TENANT, question="Welche Lenkkraft?",
    )

    assert len(calls) == 1
    logged = calls[0]
    assert logged["tenant_id"] == TENANT
    assert logged["question"] == "Welche Lenkkraft?"
    assert len(logged["config_hash"]) == 64
    assert logged["abstained"] is False
    assert logged["answer"] == result.answer
    assert set(logged["latency_ms"]) == {"retrieval_ms", "generation_ms", "total_ms"}
    assert logged["tokens"] == {"prompt": 0, "completion": 0}  # fake reports no usage
    assert logged["cost_usd"] is None  # "fake" isn't in pricing.yaml
    assert len(logged["retrieved_ids"]) == 2


async def test_generate_answer_logs_pre_generation_abstention_as_a_query(
    monkeypatch: Any,
) -> None:
    _stub_retrieval(monkeypatch, [])
    calls = _stub_query_log(monkeypatch)

    await generation.generate_answer(
        object(), object(), object(), _RefusingLlm(),
        tenant_id=TENANT, question="Wie hoch ist der Oelpreis?",
    )

    assert len(calls) == 1
    assert calls[0]["abstained"] is True
    assert calls[0]["answer"] == "NICHT_GEFUNDEN"
    assert calls[0]["retrieved_ids"] == []
    assert calls[0]["cost_usd"] is None


async def test_generate_answer_stream_writes_a_query_log_row(monkeypatch: Any) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1")])
    calls = _stub_query_log(monkeypatch)

    async for _ in generation.generate_answer_stream(
        object(), object(), object(), FakeLlmClient(),
        tenant_id=TENANT, question="Frage?",
    ):
        pass

    assert len(calls) == 1
    assert calls[0]["abstained"] is False


async def test_cost_usd_is_computed_for_a_priced_model(monkeypatch: Any) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1")])

    class _Spy:
        name = "spy"

        async def complete(
            self, *, system: str, user: str, temperature: float, max_tokens: int
        ) -> LlmResponse:
            return LlmResponse(
                text="[S1]", model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500
            )

    result = await generation.generate_answer(
        object(), object(), object(), _Spy(), tenant_id=TENANT, question="Frage?",
    )

    # gpt-4o-mini in pricing.yaml: $0.15/Mtok in, $0.60/Mtok out
    assert result.cost_usd == round(1000 / 1_000_000 * 0.15 + 500 / 1_000_000 * 0.60, 6)


async def test_cost_usd_is_none_for_an_unpriced_model(monkeypatch: Any) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1")])

    class _Spy:
        name = "spy"

        async def complete(
            self, *, system: str, user: str, temperature: float, max_tokens: int
        ) -> LlmResponse:
            return LlmResponse(
                text="[S1]", model="some-self-hosted-model",
                prompt_tokens=100, completion_tokens=50,
            )

    result = await generation.generate_answer(
        object(), object(), object(), _Spy(), tenant_id=TENANT, question="Frage?",
    )

    assert result.cost_usd is None


async def test_stream_cost_usd_uses_usage_reported_on_the_stream(monkeypatch: Any) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1")])

    class _Spy:
        name = "spy"

        async def stream(
            self, *, system: str, user: str, temperature: float, max_tokens: int
        ) -> AsyncGenerator[Delta]:
            yield Delta(text="[S1]", model="gpt-4o-mini")
            yield Delta(prompt_tokens=1000, completion_tokens=500)

    events = [
        event
        async for event in generation.generate_answer_stream(
            object(), object(), object(), _Spy(), tenant_id=TENANT, question="Frage?",
        )
    ]

    done = events[-1]
    assert done.data["cost_usd"] == round(1000 / 1_000_000 * 0.15 + 500 / 1_000_000 * 0.60, 6)
