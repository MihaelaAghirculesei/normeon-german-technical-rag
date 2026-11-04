"""add chunks.regions: the page and box of each block a chunk was cut from

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-04
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("regions", JSONB(none_as_null=True), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "regions")
