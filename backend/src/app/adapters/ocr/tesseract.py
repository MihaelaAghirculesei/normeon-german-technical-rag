"""Tesseract OCR through PyMuPDF's built-in integration: no extra Python
dependency, only the `tesseract` binary and its German language data
(`tesseract-ocr` + `tesseract-ocr-deu` in the Docker image)."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import pymupdf
import structlog

from app.core.config import settings

_log = structlog.get_logger(__name__)


class TesseractOcr:
    def __init__(self, *, tessdata: str, language: str = "deu", dpi: int = 300) -> None:
        self.tessdata = tessdata
        self.language = language
        self.dpi = dpi

    def read(self, page: Any) -> dict[str, Any]:
        textpage = page.get_textpage_ocr(
            language=self.language, dpi=self.dpi, full=True, tessdata=self.tessdata
        )
        result: dict[str, Any] = page.get_text("dict", textpage=textpage)
        return result


@lru_cache
def default_ocr() -> TesseractOcr | None:
    """The OCR engine ingestion uses, or None when OCR is switched off or
    tesseract is not installed. Without it a scanned page simply yields no
    text, as before; ingestion does not fail over it."""
    if not settings.ocr_enabled:
        return None
    try:
        tessdata = str(pymupdf.get_tessdata())  # type: ignore[no-untyped-call]
    except Exception as exc:
        _log.warning("ocr.unavailable", reason=str(exc))
        return None
    if not (Path(tessdata) / f"{settings.ocr_language}.traineddata").is_file():
        _log.warning("ocr.unavailable", reason=f"no {settings.ocr_language} data in {tessdata}")
        return None
    return TesseractOcr(tessdata=tessdata, language=settings.ocr_language, dpi=settings.ocr_dpi)
