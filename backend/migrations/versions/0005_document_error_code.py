"""add documents.error_code: why ingestion failed

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-14
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_ERROR_CODES = ("parse_timeout", "parse_failed", "embedding_failed")


def upgrade() -> None:
    # A code, not a message: the UI words it in German, logs keep the detail.
    op.add_column("documents", sa.Column("error_code", sa.Text(), nullable=True))
    allowed = ", ".join(f"'{code}'" for code in _ERROR_CODES)
    op.create_check_constraint(
        "ck_documents_error_code", "documents", f"error_code IS NULL OR error_code IN ({allowed})"
    )


def downgrade() -> None:
    op.drop_constraint("ck_documents_error_code", "documents", type_="check")
    op.drop_column("documents", "error_code")
