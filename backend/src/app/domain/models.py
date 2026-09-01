"""Pure result types shared across the retrieval pipeline. No infra
imports here -- `services/` and `api/` depend on this, never the reverse.
"""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One chunk returned by a retrieval step, with the metadata needed to
    rank it, fuse it with other rankings, and later turn it into a
    verifiable citation (filename + page + section).

    `score` is step-relative: cosine similarity for vector search, a rank
    score for FTS, a fused score after RRF. Only ordering within a single
    step's result list is meaningful.
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
