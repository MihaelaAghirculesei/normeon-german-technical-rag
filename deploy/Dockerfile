# The deployed demo: one image, one origin (ADR 0005). Built from the
# repository root: docker build -f deploy/Dockerfile .
#
# Unlike backend/Dockerfile (local development, e5 and the cross-encoder
# in-process), this image installs no torch: embeddings come from the
# Gemini API and there is no reranker, so it fits a 512 MB instance.

FROM node:22-slim AS frontend

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OCR for scanned PDF pages (Giorno 26): tesseract with German language data.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml ./
COPY backend/src ./src
# Editable, as in backend/Dockerfile: the code stays in /app/src, next to
# prompts/ and pricing.yaml. No `local` extra: no torch.
RUN pip install --upgrade pip && pip install -e .

COPY backend/prompts ./prompts
COPY backend/pricing.yaml backend/alembic.ini ./
COPY backend/migrations ./migrations
COPY deploy/start.sh ./start.sh
COPY --from=frontend /frontend/dist/frontend/browser ./static

ENV ENVIRONMENT=production \
    STATIC_DIR=/app/static \
    EMBEDDING_PROVIDER=gemini \
    EMBEDDING_MODEL=gemini-embedding-2 \
    RERANKER_PROVIDER=noop \
    OTEL_EXPORTER=none \
    RATE_LIMIT_LOGIN_PER_MINUTE=10 \
    RATE_LIMIT_QUESTIONS_PER_MINUTE=6 \
    RATE_LIMIT_UPLOADS_PER_HOUR=10

# Not root: the app needs no privileges, and a compromise should not get them.
RUN useradd --create-home --uid 1000 normeon
USER normeon

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

CMD ["sh", "start.sh"]
