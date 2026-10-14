"""users table, the normeon_app role, and row-level security

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-13

Row-level security is the second line of tenant isolation, under the
application's own `WHERE tenant_id = ...` filters: if a query ever forgets
its filter, the database still returns only the current tenant's rows.

Superusers and roles with BYPASSRLS ignore every policy, and the role the
app connects as is usually one of them (the docker image's POSTGRES_USER
is a superuser). So policies alone would protect nothing. Instead, every
tenant-scoped transaction switches to `normeon_app` -- no superuser, no
BYPASSRLS, not the table owner -- with `SET LOCAL ROLE`, and states its
tenant in `app.tenant_id` (`db/session.py`). An unset `app.tenant_id`
reads as NULL and matches no row: missing context fails closed.

`embeddings` has no tenant column and every read joins `chunks`, which is
covered; `users` and `tenants` are read before a tenant is known (login).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

APP_ROLE = "normeon_app"
TENANT_TABLES = ("documents", "chunks", "query_logs")
_TENANT_MATCH = "tenant_id = current_setting('app.tenant_id', true)::uuid"


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False
        ),
        sa.Column("email", sa.Text(), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("role IN ('admin', 'reader')", name="ck_users_role"),
    )

    # Roles are cluster-wide, so another database on the same server may
    # already have created it.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                CREATE ROLE {APP_ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS;
            END IF;
        END
        $$
        """
    )
    # The connecting role must be a member to `SET ROLE` to it (superusers
    # can anyway; a dedicated production login role cannot without this).
    op.execute(f"GRANT {APP_ROLE} TO CURRENT_USER")
    op.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
    op.execute(f"GRANT SELECT ON tenants TO {APP_ROLE}")
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON documents, chunks, embeddings TO {APP_ROLE}"
    )
    op.execute(f"GRANT SELECT, INSERT ON query_logs TO {APP_ROLE}")

    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} TO {APP_ROLE} "
            f"USING ({_TENANT_MATCH}) WITH CHECK ({_TENANT_MATCH})"
        )


def downgrade() -> None:
    for table in reversed(TENANT_TABLES):
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    op.execute(f"REVOKE ALL ON query_logs, embeddings, chunks, documents, tenants FROM {APP_ROLE}")
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {APP_ROLE}")
    # The role itself is left in place: it is cluster-wide and may be used
    # by another database on the same server.
    op.drop_table("users")
