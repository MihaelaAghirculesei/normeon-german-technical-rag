FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OCR for scanned PDF pages (Giorno 26): tesseract with German language data.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, as in CI: the default Linux wheel bundles the CUDA
# runtime (gigabytes of nvidia-* packages) and this image has no GPU.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
# Editable: the code stays in /app/src, where it finds prompts/ and
# pricing.yaml next to it. A regular install copies it into site-packages,
# where those paths do not exist for anything but uvicorn's --app-dir.
RUN pip install -e .

# Read at runtime relative to /app (services/prompts.py, services/pricing.py),
# and the migrations, so the schema can be upgraded from this image.
COPY prompts ./prompts
COPY pricing.yaml alembic.ini ./
COPY migrations ./migrations

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
