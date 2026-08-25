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
