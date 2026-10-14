# Normeon

A cited RAG assistant for German technical documentation — Lastenhefte,
UNECE regulations, manuals — that answers with verifiable citations instead
of prose that merely sounds confident.

[![CI](https://github.com/MihaelaAghirculesei/normeon-german-technical-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/MihaelaAghirculesei/normeon-german-technical-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Status: Day 25 of 30. Retrieval is feature-complete: structured German
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
Evaluation: a hand-written 50-question German eval set (10 per category
— requirement lookup, cross-reference, code lookup, unanswerable,
conflict) run for real against the ingested corpus, scored with a
two-layer correctness metric (deterministic point match + an LLM
judge), with the judge itself checked against a 15-question
human-scored sample (percent agreement 1.0, Cohen's kappa 1.0), and a
metrics module (recall/MRR/precision, citation precision, hallucinated-
citation rate, answer accuracy, correct/false abstention rate, latency
percentiles, cost per query) run for real over the full 50-question set,
and a first experiment matrix (structural vs. fixed-window chunking ×
hybrid vs. vector retrieval) reported with bootstrap confidence intervals
and paired comparisons: it found and fixed a context-selection bug worth
12.5 points of recall (0.80 → 0.925), and backs the default chunking
choice with numbers (ADR 0003, `docs/EVALUATION.md`).
Frontend: an Angular 22 app (zoneless, signals) whose API types are
generated from the backend's OpenAPI document, with CI failing on any
contract drift between the two; the document list reads live data. The
chat page streams answers over `fetch` + `ReadableStream` (POST, so not
`EventSource`), with the SSE frames themselves typed in the OpenAPI
contract. It names the pipeline phase the request is actually in
(searching, weighing sources, writing), shows the sources before the
first word of the answer, and can cancel a running request, which also
stops the model call on the server. Every `[S1]` in an answer is a
clickable chip that opens the cited passage, highlighted between the
text before and after it, with its document, version, section and page
(`GET /api/v1/chunks/{id}/context`). Documents are uploaded by drag and
drop with byte progress, then followed through ingestion (parsing,
indexing, ready) by polling. An identical file is recognised by its hash
and reported as such, and is never processed twice, even when it arrives
again while the first copy is still being processed. The document table
shows chunk counts per chunking strategy.
Security: login issues a short-lived JWT carrying the user's tenant and
role, and the tenant comes from that token only, never from the request.
Tenant isolation has two layers. The application filters every query on
the tenant, and Postgres row-level security enforces the same boundary
underneath: row-level security is the seatbelt, the application's filters
are the braces. The app switches to a dedicated non-superuser role for
every tenant-scoped transaction, because a superuser connection would
silently bypass every policy (ADR 0004). `tests/security` proves it:
- a user asking a question that only another tenant's documents can
  answer gets `NICHT_GEFUNDEN`, not a 403;
- unfiltered SQL sees one tenant only;
- cross-tenant writes are rejected;
- mutation runs show each layer holding on its own.
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

Database schema and demo users (two tenants: `anna@demo.normeon.dev`,
admin of the demo tenant that holds the corpus, and
`ben@zulieferer.normeon.dev`, reader in an empty second tenant; both with
the password from `DEMO_USER_PASSWORD`, default `normeon-demo`):

```
cd backend
alembic upgrade head
python -m app.db.seed
```

Outside development, set `ENVIRONMENT` and a random `JWT_SECRET` (32+
characters): the backend refuses to start with the default one.

Frontend (Node 22.22.3+), with the backend running:

```
cd frontend
npm ci
npm start          # http://localhost:4210, /api proxied to :8010
```

The API types in `frontend/src/app/api/schema.d.ts` are generated, never
edited by hand. After changing an endpoint: regenerate
`backend/openapi.json` (`backend/scripts/export_openapi.py`), then
`npm run gen:api` in `frontend/`. CI checks both steps.

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

## License

Code is released under the [MIT License](LICENSE).

The documents under `corpus/` are not covered by that license and keep
their own terms: `StVZO.pdf` and `FZV.pdf` are official German
regulations (amtliche Werke, § 5 UrhG — no copyright), and the
`Lastenheft-EPS-*.docx` files are synthetic samples written for this
project. See `corpus/manifest.yaml` for the provenance of each file.
