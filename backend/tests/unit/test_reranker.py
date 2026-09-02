"""Unit tests for the reranker adapters.

`CrossEncoderReranker` is exercised with a fake object swapped into
`_model`, so the real bge-reranker weights are never loaded here.
"""

import uuid
from typing import Any

from app.adapters.reranker.cross_encoder import CrossEncoderReranker
from app.adapters.reranker.noop import NoopReranker
from app.domain.models import RetrievedChunk


def _chunk(content: str, score: float = 0.0, section: str | None = None) -> RetrievedChunk:
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


# --- NoopReranker ---------------------------------------------------------


def test_noop_keeps_order_and_scores_truncating_to_top_k() -> None:
    chunks = [_chunk("a", 0.1), _chunk("b", 0.2), _chunk("c", 0.3)]

    out = NoopReranker().rerank("q", chunks, top_k=2)

    assert [c.content for c in out] == ["a", "b"]
    assert [c.score for c in out] == [0.1, 0.2]


def test_noop_returns_a_new_list() -> None:
    chunks = [_chunk("a")]
    out = NoopReranker().rerank("q", chunks, top_k=5)
    assert out == chunks
    assert out is not chunks


def test_noop_on_empty_input() -> None:
    assert NoopReranker().rerank("q", [], top_k=5) == []


# --- CrossEncoderReranker ----------------------------------------------


class _FakeCrossEncoder:
    """Returns a preset score per pair and records what it was asked."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.seen_pairs: list[tuple[str, str]] | None = None

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.seen_pairs = list(pairs)
        return self._scores


def test_cross_encoder_reorders_by_predicted_score_and_replaces_score() -> None:
    reranker = CrossEncoderReranker("unused")
    reranker._model = _FakeCrossEncoder([0.2, 0.9, 0.5])  # type: ignore[assignment]
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]

    out = reranker.rerank("welche Lenkkraft?", chunks, top_k=3)

    assert [c.content for c in out] == ["b", "c", "a"]
    assert [round(c.score, 3) for c in out] == [0.9, 0.5, 0.2]


def test_cross_encoder_builds_query_content_pairs() -> None:
    reranker = CrossEncoderReranker("unused")
    fake = _FakeCrossEncoder([0.1, 0.2])
    reranker._model = fake  # type: ignore[assignment]

    reranker.rerank("Q", [_chunk("first"), _chunk("second")], top_k=2)

    assert fake.seen_pairs == [("Q", "first"), ("Q", "second")]


def test_cross_encoder_truncates_to_top_k() -> None:
    reranker = CrossEncoderReranker("unused")
    reranker._model = _FakeCrossEncoder([0.1, 0.5, 0.9, 0.3])  # type: ignore[assignment]
    chunks = [_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d")]

    out = reranker.rerank("q", chunks, top_k=2)

    assert [c.content for c in out] == ["c", "b"]


def test_cross_encoder_on_empty_input_does_not_touch_the_model() -> None:
    reranker = CrossEncoderReranker("unused")
    sentinel: Any = object()
    reranker._model = sentinel  # type: ignore[assignment]

    assert reranker.rerank("q", [], top_k=5) == []


def test_cross_encoder_does_not_load_model_until_first_rerank() -> None:
    assert CrossEncoderReranker("unused")._model is None
