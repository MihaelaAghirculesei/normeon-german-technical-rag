from fastapi import FastAPI

from app.api.v1 import health
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(title=settings.app_name)

app.include_router(health.router)
