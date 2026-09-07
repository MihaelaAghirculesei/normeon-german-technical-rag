"""Typed shapes for the evaluation harness: the question format (plan,
Giorno 16), the run configuration the matrix varies (Giorno 20), and the
report one run writes out.

`EvalQuestion` is a Pydantic model so `backend/eval/dataset/schema.json`
can be *generated* from it (`EvalQuestion.model_json_schema()`) rather
than hand-maintained alongside it -- one source of truth, and a test
guards the checked-in file against drift. The report models are Pydantic
too, so `report.model_dump_json()` is the whole serialiser.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

CATEGORIES = (
    "requirement_lookup",
    "cross_reference",
    "code_lookup",
    "unanswerable",
    "conflict",
)
Category = Literal[
    "requirement_lookup",
    "cross_reference",
    "code_lookup",
    "unanswerable",
    "conflict",
]
Difficulty = Literal["easy", "medium", "hard"]


class GoldSource(BaseModel):
    document: str
    section: str | None = None
    page: int | None = None


class EvalQuestion(BaseModel):
    id: str = Field(pattern=r"^Q\d{3}$")
    category: Category
    question: str = Field(min_length=1)
    expected_answer_points: list[str] = Field(default_factory=list)
    gold_sources: list[GoldSource] = Field(default_factory=list)
    should_abstain: bool = False
    difficulty: Difficulty = "medium"
    notes: str | None = None


class EvalConfig(BaseModel):
    """The knobs the Week 4 matrix (Giorno 20) varies. `chunking_strategy`
    is applied by the runner today (passed straight to `generate_answer`);
    the reranker is chosen by which object the caller hands the runner;
    `retrieval_mode` and `top_k` are recorded and hashed now but only
    *applied* once Day 20 threads them through `retrieve_context` (which
    is hybrid-only, no top-k param, at time of writing)."""

    chunking_strategy: Literal["fixed_500", "structural"]
    retrieval_mode: Literal["vector", "hybrid"] = "hybrid"
    reranker: Literal["noop", "cross_encoder"]
    top_k: int = 5
    embedding_provider: str
    llm_provider: str
    llm_model: str


class CitationRecord(BaseModel):
    marker: str
    document_id: UUID
    filename: str
    page_from: int
    page_to: int
    section_path: str | None


class RetrievedRecord(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    page_from: int
    page_to: int
    section_path: str | None


class QuestionRun(BaseModel):
    question_id: str
    category: Category
    answer: str
    abstained: bool
    citations: list[CitationRecord]
    retrieved: list[RetrievedRecord]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    cost_usd: float | None
    prompt_name: str
    prompt_sha256: str
    model: str
    error: str | None = None


class EvalReport(BaseModel):
    config: EvalConfig
    config_hash: str
    created_at: datetime
    n_questions: int
    results: list[QuestionRun]
