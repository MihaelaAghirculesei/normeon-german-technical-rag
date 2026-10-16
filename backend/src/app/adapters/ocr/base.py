"""OCR for PDF pages without a text layer (Giorno 26).

A scanned page gives PyMuPDF no text, so the PDF parser hands it to a
`PageOcr`. The result has the same shape as `page.get_text("dict")`, which
lets the parser run its usual block extraction (titles, sections) over OCR
text exactly as over a text layer.
"""

from typing import Any, Protocol


class PageOcr(Protocol):
    def read(self, page: Any) -> dict[str, Any]:
        """Recognise the text on `page` (a `pymupdf.Page`) and return it in
        the structure of `page.get_text("dict")`."""
        ...
