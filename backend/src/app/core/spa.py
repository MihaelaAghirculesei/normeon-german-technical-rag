"""Serve the built Angular app from the API process, so the deployed demo is
one origin: the frontend already calls `/api/v1/...` relatively, and no
CORS or second service is needed (ADR 0005)."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response

#: Paths the SPA fallback never answers: an unknown API route stays a JSON
#: 404 instead of turning into the app's HTML.
_BACKEND_PREFIXES = ("api/", "health", "metrics")

#: Angular puts a content hash in every bundle name, so a bundle never
#: changes under its name; index.html must be revalidated to pick up a
#: new release.
_IMMUTABLE = "public, max-age=31536000, immutable"
_REVALIDATE = "no-cache"


def mount_spa(app: FastAPI, directory: Path) -> None:
    root = directory.resolve()
    index = root / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> Response:
        if path.startswith(_BACKEND_PREFIXES):
            return JSONResponse(status_code=404, content={"code": "not_found", "detail": path})
        candidate = (root / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(root):
            cache = _REVALIDATE if candidate == index else _IMMUTABLE
            return FileResponse(candidate, headers={"Cache-Control": cache})
        # A client-side route (/fragen, /dokumente, ...): the app routes it.
        return FileResponse(index, headers={"Cache-Control": _REVALIDATE})
