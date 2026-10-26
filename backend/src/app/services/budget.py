"""The daily question cap (Giorno 29): an automatic switch that stops
answering once a day's budget is spent, so a shared demo link cannot
exhaust the free-tier quota for everyone.

Every answered question -- abstentions included, they still embed the
question -- writes one `query_logs` row, so today's rows are today's
spend. The count lives in the database, not the process: it survives a
restart, and a free-tier instance restarts every time it wakes up.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import DailyLimitReachedError

_QUESTIONS_TODAY = text(
    "SELECT count(*) FROM query_logs WHERE created_at >= date_trunc('day', now())"
)


async def questions_today(session: AsyncSession) -> int:
    """All tenants' questions since midnight (database time, UTC on the
    deployment). `session` must not be tenant-scoped: under row-level
    security it would count one tenant's rows only."""
    return int(await session.scalar(_QUESTIONS_TODAY) or 0)


async def ensure_daily_budget(
    session: AsyncSession, limit: int | None, now: datetime | None = None
) -> None:
    """Raise DailyLimitReachedError when `limit` questions were asked today;
    `None` or 0 means no cap. Concurrent requests can overshoot by the
    few that check at the same moment -- a soft cap, which is enough to
    protect a quota."""
    if not limit:
        return
    if await questions_today(session) < limit:
        return
    now = now or datetime.now(UTC)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    raise DailyLimitReachedError(
        f"the demo answers {limit} questions a day; try again tomorrow",
        retry_after_s=int((midnight - now).total_seconds()) or 1,
    )
