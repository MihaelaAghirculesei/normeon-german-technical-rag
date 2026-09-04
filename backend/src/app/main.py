from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1 import chat, documents, health, search
from app.core.config import settings
from app.core.errors import NormeonError
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(title=settings.app_name)


@app.exception_handler(NormeonError)
async def normeon_error_handler(request: Request, exc: NormeonError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"code": exc.code, "detail": str(exc)})


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
