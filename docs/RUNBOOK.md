# Runbook: the deployed demo

The deployment is described in ADR 0005. This page is what to do when
something has to be put back.

## Restore the database from a nightly backup

The `backup` workflow keeps a `pg_dump` (custom format) of the demo
database as an artifact for seven days.

1. **Get the dump.** Pick a successful `backup` run and download its
   artifact:

   ```bash
   gh run list --workflow backup --limit 5
   gh run download <run-id> --dir restore
   ```

2. **Prepare an empty target database** (a new Neon database or branch,
   or a local `pgvector/pgvector:pg18` container). Use its direct
   (unpooled) connection string as `TARGET_URL`.

3. **Create the app role first.** Roles belong to the cluster, not the
   database, so `pg_dump` does not carry them -- but the dump's grants and
   row-level-security setup name `normeon_app`, and the app switches to
   it in every tenant transaction (ADR 0004). Connect as the login role
   the app will use and run:

   ```sql
   CREATE ROLE normeon_app NOLOGIN NOSUPERUSER NOBYPASSRLS;
   GRANT normeon_app TO CURRENT_USER;
   ```

4. **Restore**, with a client of the server's major version:

   ```bash
   docker run --rm -e TARGET_URL -v "$PWD/restore:/in" postgres:18 \
       sh -c 'pg_restore --no-owner --exit-on-error --dbname "$TARGET_URL" /in/normeon.dump'
   ```

   `--no-owner` makes the restoring role own every object, which is what
   the app expects; ownership in Neon cannot be assigned to another role
   anyway.

5. **Check it.** `alembic current` (with `DATABASE_URL=$TARGET_URL`) must
   print the head revision, and the row counts must match the source:

   ```sql
   SELECT (SELECT count(*) FROM documents) AS documents,
          (SELECT count(*) FROM chunks)    AS chunks,
          (SELECT count(*) FROM embeddings) AS embeddings;
   ```

6. **Point the app at it.** Set `DATABASE_URL` in the Render dashboard
   to `TARGET_URL` and redeploy. The start script runs the migrations
   (a no-op at head) and the seed (idempotent).

Without a usable dump, a fresh database is rebuilt by starting the app
against it (migrations + seed create the schema, role and demo users) and
loading the demo corpus again: see "Load the demo corpus" below.

## Load the demo corpus

The demo tenant's documents are not ingested on the server: parsing and
embedding the corpus again would spend most of a day's embedding quota
(ADR 0005). They are copied from a development database that already
holds them, with their `gemini:gemini-embedding-2` vectors
(`scripts/embed_missing.py` adds those next to e5's). Only the
structural chunks are copied: they are the ones the deployed app
retrieves, and the only ones with Gemini vectors.

1. **Keep the connection string out of the shell history and the repo.**
   Put the target's direct (unpooled) URL, as the provider gives it, in a
   file at the repository root that `*.local.env` keeps out of git:

   ```bash
   # neon.local.env
   TARGET_URL='postgresql://...?sslmode=require&channel_binding=require'
   ```

   ```bash
   set -a; . ./neon.local.env; set +a
   ```

2. **Create the schema and the demo users** by running the start
   script's two setup steps against the target, from `backend/` with
   the venv active. The URL works as given (`asyncpg_url` rewrites it):

   ```bash
   DATABASE_URL="$TARGET_URL" alembic upgrade head
   DATABASE_URL="$TARGET_URL" python -m app.db.seed
   ```

   The seed gives both tenants fixed ids, so the copied rows land in the
   demo tenant of the new database.

3. **Copy documents, chunks and vectors** from the development database
   (the compose `db` service). Its `psql` reads from the local database
   and writes to the target over TLS; ids are kept, so the chunks still
   point at their documents and parents, and the vectors at their
   chunks. `tsv` is a generated column: it is left out and the target
   computes it.

   ```bash
   docker compose exec -T -e TARGET_URL db bash -s <<'EOF'
   set -euo pipefail
   src() { psql -U normeon -d normeon -v ON_ERROR_STOP=1 -q "$@"; }
   dst() { psql "$TARGET_URL" -v ON_ERROR_STOP=1 -q "$@"; }
   demo=00000000-0000-0000-0000-000000000001
   model=gemini:gemini-embedding-2
   cols="id, document_id, tenant_id, strategy, chunk_hash, parent_id, ordinal, page_from, page_to, section_path, heading, char_start, char_end, content, content_norm, token_count, ocr"
   src -c "\copy (SELECT * FROM documents WHERE tenant_id = '$demo' AND status = 'ready') TO STDOUT" \
     | dst -c "\copy documents FROM STDIN"
   src -c "\copy (SELECT $cols FROM chunks WHERE tenant_id = '$demo' AND strategy = 'structural') TO STDOUT" \
     | dst -c "\copy chunks ($cols) FROM STDIN"
   src -c "\copy (SELECT e.* FROM embeddings e JOIN chunks c ON c.id = e.chunk_id WHERE c.tenant_id = '$demo' AND c.strategy = 'structural' AND e.model = '$model') TO STDOUT" \
     | dst -c "\copy embeddings FROM STDIN"
   EOF
   ```

   The copy runs as the database owner, which row-level security does
   not restrict (the tables do not force it). It is not idempotent: on a
   second run the primary keys collide and the script stops, so a
   partial load is cleared with `DELETE FROM documents` (it cascades to
   the chunks and their vectors) before trying again.

4. **Check it.** The counts must match the source -- for the current
   corpus 4 documents, 737 chunks and 737 embeddings:

   ```sql
   SELECT (SELECT count(*) FROM documents) AS documents,
          (SELECT count(*) FROM chunks)    AS chunks,
          (SELECT count(*) FROM embeddings) AS embeddings;
   ```

   Then log in as the demo admin and ask a question: an answer with
   citations shows that the vectors, the full-text index and the tenant
   scoping all work.

## The day's quota is spent

Two separate limits can stop answers:

- **The app's daily budget** (`DAILY_QUESTION_LIMIT`, 200): the chat
  says so and the API returns 429 until midnight UTC. Raise or clear the
  variable in the Render dashboard to lift it.
- **Gemini's free tier**: 1000 embedding requests a day per project
  (every question and every uploaded chunk counts) and a per-model
  request limit for answers. Answers then fail with a provider error
  until the quota resets at midnight Pacific time. Nothing in the app
  can lift this; wait, or use another key.
