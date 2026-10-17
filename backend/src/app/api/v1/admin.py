"""`GET /api/v1/admin/stats`: the numbers behind the frontend's statistics
page (Giorno 27). Admins only, and the caller's tenant only.

Answer statistics come from `query_logs`, so they are exact and the same
for every worker. Invented citations and LLM errors are not stored per
query; they come from this process's Prometheus counters, since it started
(`server`).
"""

from datetime import datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import Float, case, func, select

from app.api.deps import AdminDep, TenantSession
from app.core.config import settings
from app.core.metrics import sample
from app.db.models import Chunk, Document, QueryLog

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

_PHASES = ("retrieval_ms", "generation_ms", "total_ms")
_LLM_ERROR_CODES = ("llm_timeout", "llm_unavailable", "llm_misconfigured")
_RECENT = 10
_QUESTION_CHARS = 120


class Percentiles(BaseModel):
    p50: float
    p95: float


class LatencyStats(BaseModel):
    retrieval_ms: Percentiles
    generation_ms: Percentiles
    total_ms: Percentiles


class DocumentStats(BaseModel):
    ready: int
    processing: int
    failed: int
    chunks: int


class RecentQuery(BaseModel):
    created_at: datetime
    question: str
    abstained: bool
    total_ms: int
    cost_usd: float | None


class ServerCounters(BaseModel):
    citations: int
    hallucinated_citations: int
    llm_errors: int


class AdminStats(BaseModel):
    window_hours: int
    queries: int
    abstention_rate: float | None
    cost_usd_total: float
    cost_usd_per_query: float | None
    # Null while the window has no queries.
    latency_ms: LatencyStats | None
    documents: DocumentStats
    recent: list[RecentQuery]
    server: ServerCounters


@router.get("/stats", response_model=AdminStats)
async def stats(
    auth: AdminDep,
    session: TenantSession,
    hours: Annotated[int, Query(ge=1, le=24 * 7)] = 24,
) -> AdminStats:
    tenant_id = auth.tenant_id
    in_window = (QueryLog.tenant_id == tenant_id) & (
        QueryLog.created_at >= func.now() - timedelta(hours=hours)
    )

    def percentile(fraction: float, key: str) -> object:
        value = QueryLog.latency_ms[key].astext.cast(Float)
        return func.percentile_cont(fraction).within_group(value)

    row = (
        await session.execute(
            select(
                func.count(),
                func.count().filter(QueryLog.abstained),
                func.coalesce(func.sum(QueryLog.cost_usd), 0),
                *[percentile(f, key) for key in _PHASES for f in (0.5, 0.95)],
            ).where(in_window)
        )
    ).one()
    queries, abstained, cost_total = int(row[0]), int(row[1]), float(row[2] or 0)
    latency = None
    if queries:
        percentiles: tuple[Any, ...] = tuple(row[3:])
        values = [float(v) for v in percentiles]
        latency = LatencyStats(
            **{
                key: Percentiles(p50=round(values[2 * i], 1), p95=round(values[2 * i + 1], 1))
                for i, key in enumerate(_PHASES)
            }
        )

    bucket = case(
        (Document.status == "ready", "ready"),
        (Document.status == "failed", "failed"),
        else_="processing",
    ).label("bucket")
    counted = await session.execute(
        select(bucket, func.count()).where(Document.tenant_id == tenant_id).group_by(bucket)
    )
    by_status: dict[str, int] = {status: count for status, count in counted.tuples()}
    chunks = await session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.tenant_id == tenant_id)
    )

    recent = await session.scalars(
        select(QueryLog).where(in_window).order_by(QueryLog.created_at.desc()).limit(_RECENT)
    )

    return AdminStats(
        window_hours=hours,
        queries=queries,
        abstention_rate=abstained / queries if queries else None,
        cost_usd_total=round(cost_total, 6),
        cost_usd_per_query=round(cost_total / queries, 6) if queries else None,
        latency_ms=latency,
        documents=DocumentStats(
            ready=by_status.get("ready", 0),
            processing=by_status.get("processing", 0),
            failed=by_status.get("failed", 0),
            chunks=chunks or 0,
        ),
        recent=[
            RecentQuery(
                created_at=q.created_at,
                question=q.question[:_QUESTION_CHARS],
                abstained=q.abstained,
                total_ms=int(q.latency_ms.get("total_ms", 0)),
                cost_usd=float(q.cost_usd) if q.cost_usd is not None else None,
            )
            for q in recent
        ],
        server=ServerCounters(
            citations=int(sample("normeon_citations_total")),
            hallucinated_citations=int(sample("normeon_hallucinated_citations_total")),
            llm_errors=int(
                sum(
                    sample(
                        "normeon_llm_errors_total",
                        {"provider": settings.llm_provider, "code": code},
                    )
                    for code in _LLM_ERROR_CODES
                )
            ),
        ),
    )
