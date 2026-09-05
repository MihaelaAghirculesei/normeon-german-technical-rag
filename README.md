# Normeon

A cited RAG assistant for German technical documentation — Lastenhefte,
UNECE regulations, manuals — that answers with verifiable citations instead
of prose that merely sounds confident.

Status: Day 15 of 30. Retrieval is feature-complete: structured German
PDF parsing, two chunking strategies (fixed-window and structural), an
idempotent ingestion pipeline with a pluggable e5 embedding adapter,
tenant-scoped vector k-NN over an HNSW index, German full-text search
with requirement-code normalisation and a trigram fallback, hybrid
retrieval fusing the three branches with Reciprocal Rank Fusion
(ADR 0002), and a single instrumented pipeline (hybrid → cross-encoder
rerank → token-budget context selection) exposed at `POST /api/v1/search`
with per-phase timings. Generation: a versioned German answer prompt, a
context builder that assigns stable `[S1]` markers and keeps the
marker → page/section mapping backend-side, `POST /api/v1/chat` over a
thin LLM seam (OpenAI-compatible client, or a deterministic fake for
offline runs), citation validation that drops any marker the model
invents and treats an unsupported claim as an abstention rather than
surfacing it as if it were grounded, a pre-generation confidence gate
that skips the LLM call entirely below a configurable rerank score,
version-conflict detection that asks the model to flag a requirement
code with disagreeing sources, retried typed-error LLM calls so a
provider outage is a coherent HTTP response rather than a bare 500, and
`POST /api/v1/chat/stream` (SSE) that sends sources before the answer
starts streaming, with a non-cancelling heartbeat and a clean shutdown
of the model call on client disconnect, and cost tracking (a versioned
`pricing.yaml`, priced against both the streaming and non-streaming
paths) with every answer writing a `query_logs` row for tokens, cost,
per-phase latency, config fingerprint and retrieved chunks.
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

One-time setup after cloning:

```
git config core.hooksPath scripts/hooks
```

This enables a pre-push hook that blocks direct pushes to `main` and
runs the same ruff/mypy/pytest checks as CI before every push, so
nothing that would fail CI ever gets pushed. Workflow: branch → push
branch → open a PR → merge only once CI is green. Don't bypass the
hook with `--no-verify`.

## Documentation

- `docs/adr/` — Architecture Decision Records
- `docs/ARCHITECTURE.md` — system overview (coming Week 1)
- `docs/EVALUATION.md` — evaluation methodology and results (coming Week 4)
- `docs/THREAT-MODEL.md` — attacker model and mitigations (coming Week 6)
- `docs/FAILURE-MODES.md` — where this system is not reliable, with numbers (coming Week 6)
