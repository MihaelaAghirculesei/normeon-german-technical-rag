"""Unit tests for ``generate_answer`` composition. ``retrieve_context`` is
stubbed on the ``generation`` module, so no database or real models are
touched; the LLM is the FakeLlmClient (or a spy).
"""

import uuid
from typing import Any

from app.adapters.llm.base import LlmResponse
from app.adapters.llm.fake import FakeLlmClient
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
