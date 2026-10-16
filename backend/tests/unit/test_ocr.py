"""OCR fallback for scanned PDF pages (Giorno 26).

The scans are made here: a page of text is rendered to an image and that
image becomes the only content of a new page, so the page has no text
layer at all -- like a scanner's output. A fake OCR engine checks the
parser's side; the real tesseract test runs wherever tesseract and its
German data are installed (CI, the Docker image) and is skipped elsewhere.
"""

from pathlib import Path
from typing import Any

import pymupdf
import pytest

from app.adapters.ocr.tesseract import default_ocr
from app.adapters.parsing.pdf import parse_pdf

SCANNED_TEXT = "Die Lenkkraft darf 250 N nicht ueberschreiten."


def _pdf_with_a_scanned_page(tmp_path: Path, *, text_page: bool = True) -> Path:
    source = pymupdf.open()
    source.new_page().insert_text((72, 100), SCANNED_TEXT, fontsize=14)
    scan = source[0].get_pixmap(dpi=200)

    doc = pymupdf.open()
    if text_page:
        doc.new_page().insert_text(
            (72, 100), "Diese Seite hat eine echte Textebene mit genug Zeichen.", fontsize=12
        )
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=scan)
    path = tmp_path / "scan.pdf"
    doc.save(path)
    doc.close()
    source.close()
    return path


class FakeOcr:
    """Recognises a fixed sentence, in `page.get_text("dict")` form."""

    def __init__(self, text: str = SCANNED_TEXT) -> None:
        self.text = text
        self.pages: list[int] = []

    def read(self, page: Any) -> dict[str, Any]:
        self.pages.append(page.number + 1)
        if not self.text:
            return {"blocks": []}
        span = {"text": self.text, "flags": 0, "size": 11.0}
        return {"blocks": [{"type": 0, "bbox": (72, 90, 500, 110), "lines": [{"spans": [span]}]}]}


def test_a_page_without_a_text_layer_is_read_by_ocr(tmp_path: Path) -> None:
    ocr = FakeOcr()

    parsed = parse_pdf(_pdf_with_a_scanned_page(tmp_path), ocr=ocr)

    assert ocr.pages == [2]  # the text page is never sent to OCR
    assert parsed.ocr_pages == frozenset({2})
    assert [b.text for b in parsed.blocks if b.page == 2] == [SCANNED_TEXT]


def test_without_ocr_a_scanned_page_yields_no_text(tmp_path: Path) -> None:
    parsed = parse_pdf(_pdf_with_a_scanned_page(tmp_path))

    assert parsed.ocr_pages == frozenset()
    assert [b for b in parsed.blocks if b.page == 2] == []


def test_a_page_ocr_finds_nothing_on_is_not_flagged(tmp_path: Path) -> None:
    parsed = parse_pdf(_pdf_with_a_scanned_page(tmp_path), ocr=FakeOcr(text=""))

    assert parsed.ocr_pages == frozenset()


def test_the_real_tesseract_reads_a_german_scan(tmp_path: Path) -> None:
    ocr = default_ocr()
    if ocr is None:
        pytest.skip("tesseract with German language data is not installed here")

    parsed = parse_pdf(_pdf_with_a_scanned_page(tmp_path, text_page=False), ocr=ocr)

    assert parsed.ocr_pages == frozenset({1})
    recognised = " ".join(b.text for b in parsed.blocks)
    assert "Lenkkraft" in recognised
    assert "250" in recognised
