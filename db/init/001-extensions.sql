-- Runs once, on first container init, before any Alembic migration.
-- Extensions are cluster-level and versioned independently of the schema.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;
