# ADR 0005: Deploy the demo on free tiers, with hosted embeddings and no reranker

## Status
Accepted

## Context
The demo has to be reachable on a public HTTPS URL at zero cost and with
no payment card on file anywhere. That rules out every VM-based free tier
that asks for a card (Oracle Cloud, Hetzner, the hyperscalers).

What remains is small:
- **Render's free web service**: one Docker container, 512 MB RAM, 0.1
  CPU, HTTPS on `onrender.com`, spun down after 15 minutes without
  traffic, no persistent disk.
- **Neon's free Postgres**: pgvector available, 0.5 GB storage, the
  compute suspends when idle.
- **The Gemini API's free tier**: answers (`gemini-flash-lite-latest`,
  already the evaluated model) and embeddings (`gemini-embedding-2`).

The development stack does not fit in 512 MB. Local e5-large and the
cross-encoder load torch and two transformer models (~4 GB of weights
together), and one cross-encoder pair takes
~10 s on a laptop CPU (`FAILURE-MODES.md`, "Reranking + context
selection") -- at 0.1 CPU it would take minutes per question.

## Decision
1. **One container serves the API and the built Angular app**
   (`deploy/Dockerfile`, `STATIC_DIR`). One origin: no CORS, one TLS
   certificate, one service to keep awake. TLS ends at Render's proxy;
   uvicorn trusts its `X-Forwarded-For` so the per-IP limits see clients.
2. **No torch in the image.** torch and sentence-transformers moved to
   the `local` extra; the deploy image installs the base package only
   (417 MB unpacked, 153 MB compressed; 115 MB RAM after start).
3. **Embeddings from Gemini** (`gemini-embedding-2`,
   `output_dimensionality=1024`), so the `vector(1024)` column and the HNSW
   index stay as they are. Its vectors sit next to e5's in `embeddings`,
   keyed by model; switching back is a setting, not a migration.
4. **No reranker** (`RERANKER_PROVIDER=noop`). Measured, not assumed --
   see Consequences.
5. **Postgres on Neon**, connection string pasted as Neon gives it
   (libpq URL, `sslmode=require`); the app rewrites it for asyncpg.
   `start.sh` runs the migrations and the idempotent seed on every start,
   so a fresh database and a woken instance go through the same path.
6. **The free quota is protected by the app, not by hope.** Per-IP rate
   limits (10 logins/min, 6 questions/min, 10 uploads/h; 429 with
   `Retry-After`) and a daily question budget across all tenants (200,
   then 429 until midnight UTC). The frontend explains both.
7. **Backups are a GitHub Actions artifact.** A nightly `pg_dump` (custom
   format, checked with `pg_restore --list`) is kept for seven days.
   Restoring is in `docs/RUNBOOK.md`.
8. **No keep-warm ping.** A cron every 14 minutes would cost ~3000
   Actions minutes a month; the private repository has 2000. The first
   request after an idle period waits for the cold start (~30-60 s).

## Consequences
- **Retrieval got better, not worse** (E7 in `docs/EVALUATION.md`).
  Retrieval-only, structural chunks, 40 answerable questions, top 5:

  | Embeddings | Reranker | recall@5 | 95% CI | MRR |
  | --- | --- | --- | --- | --- |
  | e5-large (dev baseline) | cross-encoder | 0.925 | [0.825, 1.000] | 0.699 |
  | e5-large | none (E4) | 0.850 | | 0.696 |
  | gemini-embedding-2 | cross-encoder | 0.950 | [0.875, 1.000] | 0.726 |
  | **gemini-embedding-2 (deployed)** | **none** | **0.975** | [0.925, 1.000] | **0.790** |

  With Gemini embeddings the cross-encoder lowers recall and MRR: the
  first-stage ranking is already better than the reranker's. Dropping it
  costs nothing on this set, and saves the dependency that did not fit.
- **Answers are as good, not better.** The deployed configuration's
  answer accuracy is 0.800 [0.680, 0.900] against the baseline's 0.840
  (paired Δ -0.040, [-0.160, +0.080]: within noise), with higher
  citation precision (0.640 vs 0.500). Better retrieval does not reach
  the answers because the model over-abstains: 9 of its 10 wrong answers
  are `NICHT_GEFUNDEN`, 8 with the gold source in the context (#38).
- **The confidence gate stays uncalibrated.** `min_rerank_score_for_answer`
  is 0.0 and stops only an empty retrieval (#17); a real threshold would
  need cross-encoder scores, and fused RRF scores rank rather than
  measure relevance. The deployed system abstains through the prompt
  only.
- **The embedding quota is the binding limit.** The free tier allows
  1000 embedding requests a day per project, and every text counts --
  each question is one, each chunk of an uploaded document is one. A
  200-page upload can spend a day's quota on its own; the upload rate
  limit bounds how often that can happen, not its size. Re-embedding the
  whole corpus (737 structural chunks) does not fit in one day together
  with an evaluation run.
- **Data and requests go to Google.** Questions, retrieved passages and
  uploaded documents are sent to the Gemini API, whose free tier may use
  them to improve Google's products. The demo holds only public
  regulations and synthetic specifications. A real customer deployment
  would use a paid tier, a self-hosted embedder (the `e5_api` adapter)
  or the local stack.
- **Cold starts and suspended compute.** Both Render and Neon sleep when
  idle; the first question after a pause is slow. Acceptable for a demo,
  not for a product.
- **Single instance.** Rate-limit counters live in process memory and
  reset on restart; the daily budget is counted in `query_logs`, so it
  survives restarts.
- **Moving off the free tier** changes settings, not code: a paid Render
  instance with the `local` extra brings back e5 and the cross-encoder;
  any Postgres with pgvector replaces Neon.
