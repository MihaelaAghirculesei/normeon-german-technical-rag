"""Persist one `query_logs` row per answer (plan, Giorno 15) -- tokens
in/out, cost, per-phase latency, `config_hash`, retrieved chunk ids.
This is the write side only: the row is the audit trail the Week 4
experiment matrix reads back later, nothing in this repo reads it yet.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import QueryLog


async def record_query_log(
    session: AsyncSession,
    *,
    request_id: str,
    tenant_id: UUID,
    question: str,
    config_hash: str,
    retrieved_ids: list[UUID],
    answer: str | None,
    abstained: bool,
    latency_ms: dict[str, int],
    tokens: dict[str, int],
    cost_usd: float | None,
) -> None:
    session.add(
        QueryLog(
            request_id=request_id,
            tenant_id=tenant_id,
            question=question,
            config_hash=config_hash,
            retrieved_ids=retrieved_ids,
            answer=answer,
            abstained=abstained,
            latency_ms=latency_ms,
            tokens=tokens,
            # str(...) first: Decimal(0.1) keeps float's binary-imprecision
            # artifacts, Decimal(str(0.1)) does not.
            cost_usd=Decimal(str(cost_usd)) if cost_usd is not None else None,
        )
    )
    await session.commit()
