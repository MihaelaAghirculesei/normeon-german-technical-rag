"""A stable fingerprint of the retrieval/rerank/generation configuration
that produced one answer (plan, Giorno 15's `query_logs.config_hash`).

Two answers produced under the exact same configuration get the same
hash; anything that changed (a different reranker, a retuned `rrf_k`, a
different model) changes it. That is what lets the Week 4 experiment
matrix group and compare `query_logs` rows by "which run was this."

Pure: takes plain values, no I/O, no `settings` import -- the caller
(`services/generation.py`) decides which settings values go in, the way
`domain/context.build_context` and `domain/citations.
extract_and_validate` already keep this module free of anything but the
values it's handed.
"""

from __future__ import annotations

import hashlib
import json


def compute_config_hash(**fields: object) -> str:
    """sha256 of `fields`, canonicalised (sorted keys, compact
    separators) so equal configurations hash identically regardless of
    keyword-argument order."""
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
