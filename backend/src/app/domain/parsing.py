from dataclasses import dataclass
from typing import Literal

BlockKind = Literal["text", "table"]


@dataclass(frozen=True)
class Block:
    text: str
    page: int
    bbox: tuple[float, float, float, float]
    font_size: float
    bold: bool
    kind: BlockKind = "text"
    is_title: bool = False
    section_path: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    blocks: list[Block]
    # Pages whose text came from OCR because they carried (almost) no text
    # layer -- scans. Their text may contain recognition errors, so chunks
    # from them are flagged all the way to the UI (Giorno 26).
    ocr_pages: frozenset[int] = frozenset()
