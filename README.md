# Normeon

A cited RAG assistant for German technical documentation — Lastenhefte,
UNECE regulations, manuals — that answers with verifiable citations instead
of prose that merely sounds confident.

Status: Day 1 of 30. Backend skeleton, health check, corpus sourcing.
This README will be rewritten on delivery day with a demo GIF, results,
and links to the failure-mode catalogue.

## Stack

Angular 22 · FastAPI · PostgreSQL 18 + pgvector · hybrid search (vector +
full-text) · cross-encoder reranking · a 50-question evaluation suite.

## Scope

| In scope | Out of scope |
|---|---|
| PDF/DOCX upload, idempotent ingestion | Editing documents |
| Structural + fixed-window chunking, compared experimentally | Fine-tuning models |
| Hybrid retrieval + reranking | Multi-hop / agentic retrieval |
| Backend-validated citations | Generating new documents |
| SSE streaming | Voice, images, multimodal input |
| Multi-tenant with Row-Level Security | Enterprise SSO, SCIM |
| 50-question eval suite + experiment matrix | Mobile app |
| Single-node deploy | Autoscaling, HA |

## Local development

```
docker compose up
curl localhost:8010/health
```

(Ports are offset from the defaults — 8010/5433/8081 — to avoid clashing
with other local projects. See `docker-compose.yml`.)

## Documentation

- `docs/adr/` — Architecture Decision Records
- `docs/ARCHITECTURE.md` — system overview (coming Week 1)
- `docs/EVALUATION.md` — evaluation methodology and results (coming Week 4)
- `docs/THREAT-MODEL.md` — attacker model and mitigations (coming Week 6)
- `docs/FAILURE-MODES.md` — where this system is not reliable, with numbers (coming Week 6)
