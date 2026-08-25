# ADR 0001 — PostgreSQL + pgvector instead of a dedicated vector database

## Status
Accepted

## Context
Normeon needs vector similarity search over document chunks, alongside
relational data (tenants, documents, query logs) and full-text search over
German text. The corpus target is 400-600 pages (a few thousand chunks),
not billions of vectors.

## Decision
Use PostgreSQL 18 with the pgvector extension (HNSW index) as the single
datastore, instead of a dedicated vector database (Weaviate, Qdrant, ...).

## Reasoning
- One database means one transaction boundary: a chunk, its embedding, and
  its tenant ownership are written and read consistently, with no
  dual-write or eventual-consistency problem between two systems.
- Row-Level Security gives tenant isolation enforced by the database
  itself, not by application code remembering to filter correctly.
- PostgreSQL full-text search (tsvector/GIN) and pg_trgm live in the same
  engine, which is exactly what hybrid retrieval needs — one query can
  combine a vector branch and a lexical branch against the same tables.
- At this corpus size, HNSW in pgvector is fast enough (sub-150ms) that a
  dedicated vector database would add operational surface without a
  measurable benefit.

## Consequences
- If the corpus grows into the billions of vectors, or if sub-millisecond
  ANN latency becomes a hard requirement, this decision should be
  revisited — that is a real limit of this choice, not a hypothetical one.
- Index maintenance (HNSW build time, `ef_search` tuning) becomes the
  team's responsibility instead of a managed vector-DB feature.
