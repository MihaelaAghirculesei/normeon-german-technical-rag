"""add HNSW index on embeddings.vec

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-26

Deliberately separate from 0001: an HNSW index built on an empty table is
wasted work and distorts the first bulk-load timings. This runs after the
corpus has been ingested for the first time (Day 5).

Cosine distance (`vector_cosine_ops`) matches the e5 convention of
L2-normalised embeddings compared by cosine similarity. m / ef_construction
are pgvector's defaults, adequate for a corpus of this size; revisit above
roughly 50k chunks.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX idx_emb_hnsw ON embeddings "
        "USING hnsw (vec vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_emb_hnsw")
