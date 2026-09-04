"""Unit tests for ``generate_answer`` composition. ``retrieve_context`` is
stubbed on the ``generation`` module, so no database or real models are
touched; the LLM is the FakeLlmClient (or a spy).
"""

import uuid
from typing import Any

from app.adapters.llm.base import LlmResponse
from app.adapters.llm.fake import FakeLlmClient
from app.domain.models import PipelineTiming, RetrievalResult, RetrievedChunk
from app.services import generation

TENANT = uuid.uuid4()


def _chunk(section: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="StVZO.pdf",
        content=f"Regelungstext zu Abschnitt {section}.",
        page_from=12,
        page_to=12,
        section_path=section,
        heading=None,
        score=0.8,
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
    return seen


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


async def test_empty_retrieval_yields_no_sources_and_nicht_gefunden(
    monkeypatch: Any,
) -> None:
    _stub_retrieval(monkeypatch, [])

    # With no chunks the context block is empty, which the prompt tells the
    # model to answer with NICHT_GEFUNDEN. FakeLlmClient scans the whole
    # rendered prompt for [S..] markers and would "cite" the ones in the
    # instructions, so pin its reply to what a real model returns here.
    result = await generation.generate_answer(
        object(), object(), object(), FakeLlmClient(canned="NICHT_GEFUNDEN"),
        tenant_id=TENANT, question="Wie hoch ist der Oelpreis?",
    )

    assert result.sources == []
    assert result.answer == "NICHT_GEFUNDEN"


async def test_prompt_name_override_is_honoured(monkeypatch: Any) -> None:
    _stub_retrieval(monkeypatch, [_chunk("1")])

    result = await generation.generate_answer(
        object(), object(), object(), FakeLlmClient(),
        tenant_id=TENANT, question="q", prompt_name="answer_de.v1",
    )

    assert result.prompt_name == "answer_de.v1"
