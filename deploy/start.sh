#!/bin/sh
# Start the deployed app. Both setup steps are idempotent, so every
# restart -- including a free-tier instance waking up -- runs them safely.
set -eu

# The schema, row-level security and the app role.
alembic upgrade head
# The two demo tenants and their users (README lists the credentials).
python -m app.db.seed

# The platform sets PORT and terminates TLS in front of the app; trust its
# X-Forwarded-For so the per-IP rate limit sees the client, not the proxy.
exec uvicorn app.main:app \
    --app-dir src \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --proxy-headers \
    --forwarded-allow-ips "*"
