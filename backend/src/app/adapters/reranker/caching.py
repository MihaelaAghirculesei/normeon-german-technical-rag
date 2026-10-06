"""A `Reranker` decorator that remembers every (query, chunk) score in a
small SQLite file, for the evaluation harness.

A cross-encoder score depends only on the model, its truncation length,
the query and the chunk text -- not on which retrieval mode or top-k
produced the candidate. The experiment matrix re-asks the same 50
questions under many configurations whose candidate sets overlap heavily
(vector vs. hybrid over one chunking strategy; the top-k, LLM and
abstention-threshold ablations rerank *identical* candidates), and on CPU
one pair costs seconds. Persisting scores also lets a run that dies
halfway resume without re-scoring what it already paid for.

Not used on the serving path: a live request reranks fresh.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

from app.adapters.reranker.base import Reranker
from app.domain.models import RetrievedChunk


class CachingReranker:
    """Wraps `inner`, scoring through it only the pairs not yet cached.

    `namespace` must change whenever the score itself would (model name,
    `max_length`, ...): it is part of every cache key, so one file can
    hold several models' scores without collisions. Ordering and ties are
    identical to calling `inner` directly -- candidates are sorted by
    score, stable on their incoming order.
    """

    def __init__(self, inner: Reranker, cache_path: Path, *, namespace: str) -> None:
        self._inner = inner
        self._namespace = namespace
        self.name = inner.name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # `rerank` runs off the event loop via `asyncio.to_thread`, so
        # the connection is shared across worker threads behind a lock.
        self._conn = sqlite3.connect(cache_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS rerank_scores (key TEXT PRIMARY KEY, score REAL NOT NULL)"
        )
        self._conn.commit()
        self._lock = threading.Lock()

    def _key(self, query: str, content: str) -> str:
        raw = "\0".join((self._namespace, query, content)).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _lookup(self, keys: list[str]) -> dict[str, float]:
        unique = list(dict.fromkeys(keys))
        placeholders = ",".join("?" * len(unique))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT key, score FROM rerank_scores WHERE key IN ({placeholders})",  # noqa: S608
                unique,
            ).fetchall()
        return {key: float(score) for key, score in rows}

    def _store(self, scores: dict[str, float]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO rerank_scores (key, score) VALUES (?, ?)",
                list(scores.items()),
            )
            self._conn.commit()

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        keys = [self._key(query, chunk.content) for chunk in chunks]
        scores = self._lookup(keys)

        missing: dict[str, RetrievedChunk] = {}
        for key, chunk in zip(keys, chunks, strict=True):
            if key not in scores:
                missing.setdefault(key, chunk)
        if missing:
            fresh = self._inner.rerank(query, list(missing.values()), len(missing))
            new_scores = {self._key(query, chunk.content): chunk.score for chunk in fresh}
            self._store(new_scores)
            scores.update(new_scores)

        ranked = sorted(
            zip(chunks, keys, strict=True), key=lambda pair: scores[pair[1]], reverse=True
        )
        return [replace(chunk, score=scores[key]) for chunk, key in ranked[:top_k]]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
