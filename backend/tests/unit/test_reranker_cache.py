"""Unit tests for `CachingReranker`, wrapped around a counting fake so
every assertion can check *which* pairs actually reached the inner
reranker."""

import uuid
from dataclasses import replace
from pathlib import Path

from app.adapters.reranker.caching import CachingReranker
from app.domain.models import RetrievedChunk


def _chunk(content: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="StVZO.pdf",
        content=content,
        page_from=1,
        page_to=1,
        section_path=None,
        heading=None,
        score=score,
    )


class _CountingReranker:
    """Scores a chunk by a fixed table keyed on its content and records
    every (query, content) pair it was asked to score."""

    name = "cross_encoder"

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, str]] = []

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        self.calls.extend((query, c.content) for c in chunks)
        scored = [replace(c, score=self._scores[c.content]) for c in chunks]
        return sorted(scored, key=lambda c: c.score, reverse=True)[:top_k]


SCORES = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.7}


def test_matches_the_inner_reranker_on_a_cold_cache(tmp_path: Path) -> None:
    inner = _CountingReranker(SCORES)
    cached = CachingReranker(inner, tmp_path / "c.sqlite3", namespace="m|512")
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]

    out = cached.rerank("q", chunks, top_k=2)

    assert [c.content for c in out] == ["b", "c"]
    assert [c.score for c in out] == [0.9, 0.5]
    assert [c.chunk_id for c in out] == [chunks[1].chunk_id, chunks[2].chunk_id]
    assert cached.name == "cross_encoder"


def test_only_unseen_pairs_reach_the_inner_reranker(tmp_path: Path) -> None:
    inner = _CountingReranker(SCORES)
    cached = CachingReranker(inner, tmp_path / "c.sqlite3", namespace="m|512")
    cached.rerank("q", [_chunk("a"), _chunk("b")], top_k=5)
    inner.calls.clear()

    out = cached.rerank("q", [_chunk("b"), _chunk("d"), _chunk("a")], top_k=5)

    assert inner.calls == [("q", "d")]
    assert [c.content for c in out] == ["b", "d", "a"]


def test_scores_survive_a_new_instance_on_the_same_file(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite3"
    first = CachingReranker(_CountingReranker(SCORES), path, namespace="m|512")
    first.rerank("q", [_chunk("a"), _chunk("b")], top_k=5)
    first.close()

    inner = _CountingReranker(SCORES)
    second = CachingReranker(inner, path, namespace="m|512")
    out = second.rerank("q", [_chunk("a"), _chunk("b")], top_k=5)

    assert inner.calls == []
    assert [c.score for c in out] == [0.9, 0.1]


def test_a_different_namespace_or_query_is_a_miss(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite3"
    CachingReranker(_CountingReranker(SCORES), path, namespace="m|512").rerank(
        "q", [_chunk("a")], top_k=5
    )

    inner = _CountingReranker(SCORES)
    other_model = CachingReranker(inner, path, namespace="m|1024")
    other_model.rerank("q", [_chunk("a")], top_k=5)
    other_model.rerank("another q", [_chunk("a")], top_k=5)

    assert inner.calls == [("q", "a"), ("another q", "a")]


def test_ties_keep_the_incoming_order(tmp_path: Path) -> None:
    inner = _CountingReranker({"x": 0.5, "y": 0.5, "z": 0.5})
    cached = CachingReranker(inner, tmp_path / "c.sqlite3", namespace="m")

    out = cached.rerank("q", [_chunk("y"), _chunk("z"), _chunk("x")], top_k=3)

    assert [c.content for c in out] == ["y", "z", "x"]


def test_duplicate_content_is_scored_once(tmp_path: Path) -> None:
    inner = _CountingReranker(SCORES)
    cached = CachingReranker(inner, tmp_path / "c.sqlite3", namespace="m")

    out = cached.rerank("q", [_chunk("a"), _chunk("a")], top_k=5)

    assert inner.calls == [("q", "a")]
    assert len(out) == 2


def test_empty_input_touches_nothing(tmp_path: Path) -> None:
    inner = _CountingReranker(SCORES)
    cached = CachingReranker(inner, tmp_path / "c.sqlite3", namespace="m")

    assert cached.rerank("q", [], top_k=5) == []
    assert inner.calls == []
