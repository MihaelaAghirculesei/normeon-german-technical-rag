from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import metrics
from app.api.v1 import admin, auth, chat, chunks, documents, health, search
from app.core.config import insecure_settings, settings
from app.core.errors import NormeonError
from app.core.limits import BodySizeLimit
from app.core.logging import configure_logging
from app.core.observability import configure_tracing
from app.core.request_context import RequestIdMiddleware
from app.core.spa import mount_spa

configure_logging()

# Refuse to start with laptop-only settings (the default JWT secret) in any
# environment that is not development or test.
_problems = insecure_settings(settings)
if _problems:
    raise RuntimeError("insecure configuration: " + "; ".join(_problems))

app = FastAPI(title=settings.app_name)
# Oversized bodies are refused from the headers, before they are read.
# The upload limit leaves room for the multipart framing around the file.
app.add_middleware(
    BodySizeLimit,
    max_upload_bytes=settings.max_upload_mb * 2**20 + 64 * 2**10,
    max_request_bytes=settings.max_request_kb * 2**10,
)
# Added last, so outermost: even a refused upload gets a request id.
app.add_middleware(RequestIdMiddleware)
configure_tracing(app)


@app.exception_handler(NormeonError)
async def normeon_error_handler(request: Request, exc: NormeonError) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "detail": str(exc)},
        headers=headers,
    )


app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(chunks.router)
app.include_router(admin.router)
# Last: its catch-all route must not shadow any API route.
if settings.static_dir is not None:
    mount_spa(app, settings.static_dir)
