from typing import Protocol

from app.domain.models import RetrievedChunk


class Reranker(Protocol):
    """Re-scores already-retrieved chunks by looking at the (query, chunk)
    pair *jointly*, which the bi-encoder retrievers cannot -- they embed
    the two sides independently. Given the hybrid candidates it returns
    them re-sorted best-first, truncated to `top_k`, with `.score`
    replaced by the reranker's own score.

    `rerank` is synchronous and CPU-bound (a cross-encoder forward pass);
    callers run it off the event loop with `asyncio.to_thread`, the same
    way the embedders are called.

    `scores_relevance` says whether the scores it returns measure how
    relevant a chunk is to the query, on a fixed scale a threshold can be
    set against. A cross-encoder's do; a reranker that only keeps the
    incoming order passes on fused RRF scores, which say where a chunk
    ranked, not how relevant it is -- the confidence gate ignores those.
    """

    name: str
    scores_relevance: bool

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...
