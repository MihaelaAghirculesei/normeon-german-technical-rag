from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from app.api.deps import DbSession

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    db: str
    pgvector: str


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health(session: DbSession) -> HealthResponse:
    db_version = await session.scalar(text("SHOW server_version"))
    pgvector_version = await session.scalar(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    )
    return HealthResponse(
        status="ok",
        db=db_version or "unknown",
        pgvector=pgvector_version or "not_installed",
    )
