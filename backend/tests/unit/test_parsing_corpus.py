"""Table extraction on the real corpus. Synthetic test PDFs do not show how a
PyMuPDF upgrade changes the text of real tables: 2.0 glued their words
together while every other test stayed green."""

from pathlib import Path

import pymupdf

from app.adapters.parsing.pdf import _extract_tables

CORPUS = Path(__file__).parents[3] / "corpus"


def _table_text(document: str, page: int) -> str:
    doc = pymupdf.open(CORPUS / document)  # type: ignore[no-untyped-call]
    try:
        _, blocks = _extract_tables(doc[page - 1], page)
    finally:
        doc.close()  # type: ignore[no-untyped-call]
    return "\n".join(block.text for block in blocks)


def test_table_cells_keep_the_spaces_between_words() -> None:
    assert "Dienstfahrzeuge der auf Grund des Nordatlantikvertrages" in _table_text("FZV.pdf", 80)


def test_table_cells_keep_section_numbers_intact() -> None:
    assert "2.1.1" in _table_text("StVZO.pdf", 89)
