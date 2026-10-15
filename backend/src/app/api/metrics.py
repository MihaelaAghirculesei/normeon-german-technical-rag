"""`GET /metrics`: the Prometheus exposition of `core.metrics` (Giorno 27).

Outside `/api/v1` and the OpenAPI document: it is for a scraper, not for
the frontend. Open by default, as a scraper on the internal network
expects; with `settings.metrics_token` set it needs that bearer token.
"""

import hmac

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.config import settings
from app.core.errors import NotAuthenticatedError

router = APIRouter(include_in_schema=False)


@router.get("/metrics")
def metrics(request: Request) -> Response:
    if settings.metrics_token:
        offered = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(offered.encode(), settings.metrics_token.encode()):
            raise NotAuthenticatedError("metrics need the metrics token")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
