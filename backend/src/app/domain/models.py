"""Pure result types shared across the retrieval pipeline. No infra
imports here -- `services/` and `api/` depend on this, never the reverse.
"""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

# The first retrieval phase: hybrid (vector + full-text + trigram, fused
# by RRF) by default; one branch alone for the evaluation matrix.
RetrievalMode = Literal["vector", "fts", "hybrid"]

# Which of the question's terms a full-text match needs: all of them (the
# default, `websearch_to_tsquery`), or any one, ranked by how many match.
FtsMatch = Literal["all", "any"]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One chunk returned by a retrieval step, with the metadata needed to
    rank it, fuse it with other rankings, and later turn it into a
    verifiable citation (filename + page + section).

    `score` is step-relative: cosine similarity for vector search, a rank
    score for FTS, a fused score after RRF, a cross-encoder score after
    reranking. Only ordering within a single step's result list is
    meaningful.
    """

    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    page_from: int
    page_to: int
    section_path: str | None
    heading: str | None
    score: float
    version_label: str | None = None
    ocr: bool = False  # text recognised by OCR from a scanned page


@dataclass(frozen=True, slots=True)
class PipelineTiming:
    """Wall-clock milliseconds for each phase of `retrieve_context`, so a
    caller (and the debug endpoint) can see where the time went."""

    hybrid_ms: float
    rerank_ms: float
    select_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Output of the full retrieval pipeline.

    `context` is the final token-budgeted selection, best-first.
    `reranked` is everything the reranker returned before budget
    selection -- kept for inspection and offline evaluation.
    """

    context: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    timing: PipelineTiming
