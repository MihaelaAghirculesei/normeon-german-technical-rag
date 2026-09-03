"""Retrieval pipeline.

Split by concern; this package re-exports the flat public surface so
callers keep using `from app.services.retrieval import X`.

- `vector`   -- vector k-NN (`vector_search`, `vector_search_by_vector`)
- `lexical`  -- German full-text + trigram (`fts_search`; the
                `_fts_branch` / `_trgm_branch` helpers `hybrid` reuses)
- `hybrid`   -- RRF fusion of the three branches (`hybrid_search`)
- `pipeline` -- rerank + token-budget context (`retrieve_context`,
                `_select_context`)

Tests that need to monkeypatch a function patch it on its own submodule
(e.g. `retrieval.pipeline.hybrid_search`), not here.
"""

from app.services.retrieval.hybrid import hybrid_search
from app.services.retrieval.lexical import _fts_branch, _trgm_branch, fts_search
from app.services.retrieval.pipeline import _select_context, retrieve_context
from app.services.retrieval.vector import vector_search, vector_search_by_vector

__all__ = [
    "_fts_branch",
    "_select_context",
    "_trgm_branch",
    "fts_search",
    "hybrid_search",
    "retrieve_context",
    "vector_search",
    "vector_search_by_vector",
]
