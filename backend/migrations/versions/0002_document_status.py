"""add documents.status for ingestion progress tracking

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_STATUSES = ("pending", "parsing", "embedding", "ready", "failed")


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
    )
    allowed = ", ".join(f"'{status}'" for status in _STATUSES)
    op.create_check_constraint("ck_documents_status", "documents", f"status IN ({allowed})")


def downgrade() -> None:
    op.drop_constraint("ck_documents_status", "documents", type_="check")
    op.drop_column("documents", "status")
