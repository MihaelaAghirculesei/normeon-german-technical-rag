"""Unit tests for the context-selection step and the `retrieve_context`
composition. `hybrid_search` is stubbed so no database is needed.
"""

import uuid
from typing import Any

import pytest

from app.domain.models import RetrievedChunk
from app.services import retrieval
from app.services.retrieval import _select_context, retrieve_context

TENANT = uuid.uuid4()


def _chunk(content: str, section: str | None = None, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="StVZO.pdf",
        content=content,
        page_from=1,
        page_to=1,
        section_path=section,
        heading=None,
        score=score,
    )


# --- _select_context ---------------------------------------------------


def test_select_stops_before_exceeding_the_token_budget() -> None:
    chunks = [_chunk("one two three"), _chunk("four five six"), _chunk("seven eight nine")]

    selected = _select_context(chunks, token_budget=6)

    assert [c.content for c in selected] == ["one two three", "four five six"]


def test_select_always_returns_at_least_the_top_chunk() -> None:
    huge = _chunk(" ".join(["w"] * 100))

    selected = _select_context([huge, _chunk("small")], token_budget=10)

    assert selected == [huge]


def test_select_skips_a_section_path_already_represented() -> None:
    chunks = [
        _chunk("first from A", section="1 > 1.1"),
        _chunk("second from A", section="1 > 1.1"),
        _chunk("from B", section="1 > 1.2"),
    ]

    selected = _select_context(chunks, token_budget=1000)

    assert [c.content for c in selected] == ["first from A", "from B"]


def test_select_never_dedups_chunks_without_a_section_path() -> None:
    chunks = [_chunk("a", section=None), _chunk("b", section=None), _chunk("c", section=None)]

    selected = _select_context(chunks, token_budget=1000)

    assert [c.content for c in selected] == ["a", "b", "c"]


def test_select_on_empty_input() -> None:
    assert _select_context([], token_budget=1000) == []


def test_select_keeps_a_later_chunk_that_still_fits_after_a_skip() -> None:
    # middle chunk is a section dup and skipped; the third still fits
    chunks = [
        _chunk("aa bb", section="s1"),
        _chunk("cc dd ee ff", section="s1"),
        _chunk("gg hh", section="s2"),
    ]

    selected = _select_context(chunks, token_budget=4)

    assert [c.content for c in selected] == ["aa bb", "gg hh"]


# --- retrieve_context ------------------------------------------------------


class _RecordingReranker:
    name = "recording"

    def __init__(self) -> None:
        self.seen: tuple[str, int] | None = None

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        self.seen = (query, top_k)
        return list(reversed(chunks))[:top_k]


@pytest.fixture
def stub_hybrid(monkeypatch: pytest.MonkeyPatch) -> list[RetrievedChunk]:
    hits = [_chunk("alpha", section="s1"), _chunk("beta", section="s2"), _chunk("gamma")]

    async def fake_hybrid_search(
        session: Any, embedder: Any, **kwargs: Any
    ) -> list[RetrievedChunk]:
        fake_hybrid_search.kwargs = kwargs  # type: ignore[attr-defined]
        return hits

    monkeypatch.setattr(retrieval, "hybrid_search", fake_hybrid_search)
    return hits


async def test_retrieve_context_runs_hybrid_then_rerank_then_select(
    stub_hybrid: list[RetrievedChunk],
) -> None:
    reranker = _RecordingReranker()

    result = await retrieve_context(
        object(), object(), reranker, tenant_id=TENANT, question="Q", rerank_top_k=2
    )

    assert reranker.seen == ("Q", 2)
    # reranker reversed [alpha, beta, gamma] and kept 2 -> [gamma, beta]
    assert [c.content for c in result.reranked] == ["gamma", "beta"]
    assert [c.content for c in result.context] == ["gamma", "beta"]


async def test_retrieve_context_reports_per_phase_timings(
    stub_hybrid: list[RetrievedChunk],
) -> None:
    result = await retrieve_context(
        object(), object(), _RecordingReranker(), tenant_id=TENANT, question="Q"
    )

    t = result.timing
    assert t.hybrid_ms >= 0 and t.rerank_ms >= 0 and t.select_ms >= 0
    assert t.total_ms >= t.hybrid_ms + t.rerank_ms + t.select_ms - 1e-6


async def test_retrieve_context_applies_the_token_budget_after_rerank(
    stub_hybrid: list[RetrievedChunk],
) -> None:
    # reranker reverses to [gamma, beta, alpha] (each one word); budget 2 fits gamma + beta
    result = await retrieve_context(
        object(),
        object(),
        _RecordingReranker(),
        tenant_id=TENANT,
        question="Q",
        rerank_top_k=3,
        token_budget=2,
    )

    assert [c.content for c in result.context] == ["gamma", "beta"]
    assert [c.content for c in result.reranked] == ["gamma", "beta", "alpha"]


async def test_retrieve_context_forwards_strategy_and_candidate_k_to_hybrid(
    stub_hybrid: list[RetrievedChunk],
) -> None:
    await retrieve_context(
        object(),
        object(),
        _RecordingReranker(),
        tenant_id=TENANT,
        question="Q",
        strategy="fixed_500",
        candidate_k=25,
    )

    kwargs = retrieval.hybrid_search.kwargs  # type: ignore[attr-defined]
    assert kwargs["strategy"] == "fixed_500"
    assert kwargs["candidate_k"] == 25
    assert kwargs["tenant_id"] == TENANT
