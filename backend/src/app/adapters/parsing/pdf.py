import re
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import pymupdf

from app.domain.parsing import Block, ParsedDocument

BBox = tuple[float, float, float, float]

_NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\s")
_PARAGRAPH_HEADING_RE = re.compile(r"^§\s*\d+[a-z]?")
_BOLD_FLAG = 1 << 4  # PyMuPDF span flags: bit 4 marks a bold span

_HEADER_FOOTER_MIN_PAGES = 2
_HEADER_FOOTER_RATIO = 0.6
_BBOX_GROUPING_TOLERANCE = 5.0  # points; groups a header/footer's anchor position across pages
_BBOX_CONTAINMENT_TOLERANCE = 2.0  # points; slack when checking a text block sits inside a table


def parse_pdf(path: str | Path) -> ParsedDocument:
    doc = pymupdf.open(path)  # type: ignore[no-untyped-call]
    try:
        page_count = doc.page_count
        blocks: list[Block] = []
        for page_index in range(page_count):
            page = doc[page_index]
            page_num = page_index + 1
            table_bboxes, table_blocks = _extract_tables(page, page_num)
            text_blocks = _extract_text_blocks(page, page_num, table_bboxes)
            page_blocks = text_blocks + table_blocks
            page_blocks.sort(key=lambda b: b.bbox[1])
            blocks.extend(page_blocks)
    finally:
        doc.close()  # type: ignore[no-untyped-call]

    blocks = _filter_repeated_headers_footers(blocks, page_count)
    blocks = _detect_titles(blocks)
    blocks = _assign_section_paths(blocks)
    return ParsedDocument(blocks=blocks)


def _extract_text_blocks(
    page: Any, page_num: int, exclude_bboxes: list[BBox]
) -> list[Block]:
    result: list[Block] = []
    for raw_block in page.get_text("dict")["blocks"]:
        if raw_block.get("type") != 0:
            continue
        spans = [s for line in raw_block.get("lines", []) for s in line["spans"]]
        if not spans:
            continue
        text = "".join(s["text"] for s in spans).strip()
        if not text:
            continue
        bbox: BBox = tuple(raw_block["bbox"])
        if any(_bbox_inside(table_bbox, bbox) for table_bbox in exclude_bboxes):
            continue
        bold = any(s["flags"] & _BOLD_FLAG for s in spans)
        result.append(
            Block(text=text, page=page_num, bbox=bbox, font_size=spans[0]["size"], bold=bold)
        )
    return result


def _extract_tables(page: Any, page_num: int) -> tuple[list[BBox], list[Block]]:
    bboxes: list[BBox] = []
    blocks: list[Block] = []
    for table in page.find_tables().tables:
        bbox: BBox = tuple(table.bbox)
        bboxes.append(bbox)
        markdown = _table_to_markdown(table.extract())
        if not markdown:
            continue
        blocks.append(
            Block(text=markdown, page=page_num, bbox=bbox, font_size=0.0, bold=False, kind="table")
        )
    return bboxes, blocks


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    if not rows:
        return ""

    def clean(cell: str | None) -> str:
        if cell is None:
            return ""
        return " / ".join(part.strip() for part in cell.split("\n") if part.strip())

    header, *body = [[clean(cell) for cell in row] for row in rows]
    col_count = len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in body:
        padded = row + [""] * (col_count - len(row))
        lines.append("| " + " | ".join(padded[:col_count]) + " |")
    return "\n".join(lines)


def _bbox_inside(outer: BBox, inner: BBox, tol: float = _BBOX_CONTAINMENT_TOLERANCE) -> bool:
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return ix0 >= ox0 - tol and iy0 >= oy0 - tol and ix1 <= ox1 + tol and iy1 <= oy1 + tol


def _filter_repeated_headers_footers(blocks: list[Block], page_count: int) -> list[Block]:
    if page_count < _HEADER_FOOTER_MIN_PAGES:
        return blocks

    def anchor(b: Block) -> tuple[float, float]:
        return (
            round(b.bbox[0] / _BBOX_GROUPING_TOLERANCE),
            round(b.bbox[1] / _BBOX_GROUPING_TOLERANCE),
        )

    pages_by_anchor: dict[tuple[float, float], set[int]] = {}
    for b in blocks:
        pages_by_anchor.setdefault(anchor(b), set()).add(b.page)

    repeated = {
        key
        for key, pages in pages_by_anchor.items()
        if len(pages) >= _HEADER_FOOTER_MIN_PAGES and len(pages) / page_count > _HEADER_FOOTER_RATIO
    }
    return [b for b in blocks if anchor(b) not in repeated]


def _detect_titles(blocks: list[Block]) -> list[Block]:
    body_sizes = [b.font_size for b in blocks if b.kind == "text" and b.font_size > 0]
    median_size = statistics.median(body_sizes) if body_sizes else 0.0

    result: list[Block] = []
    for b in blocks:
        if b.kind != "text":
            result.append(b)
            continue
        is_title = (
            b.font_size > median_size
            or bool(_NUMBERED_HEADING_RE.match(b.text))
            or bool(_PARAGRAPH_HEADING_RE.match(b.text))
        )
        result.append(replace(b, is_title=is_title))
    return result


def _extract_section_code(text: str) -> str | None:
    match = _PARAGRAPH_HEADING_RE.match(text)
    if match:
        return re.sub(r"\s+", "", match.group(0))
    match = _NUMBERED_HEADING_RE.match(text)
    if match:
        return text[: match.end()].strip()
    return None


def _assign_section_paths(blocks: list[Block]) -> list[Block]:
    current_path: str | None = None
    result: list[Block] = []
    for b in blocks:
        if b.is_title:
            code = _extract_section_code(b.text)
            if code is not None:
                current_path = code
        result.append(replace(b, section_path=current_path))
    return result
