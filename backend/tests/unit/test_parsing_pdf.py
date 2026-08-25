from collections.abc import Sequence
from pathlib import Path

import pymupdf

from app.adapters.parsing.pdf import _table_to_markdown, parse_pdf

PageContent = Sequence[tuple[tuple[float, float], str, float, str]]


def _make_pdf(tmp_path: Path, pages: Sequence[PageContent]) -> Path:
    doc = pymupdf.open()
    for _ in pages:
        doc.new_page()
    for i, items in enumerate(pages):
        page = doc[i]
        for point, text, fontsize, fontname in items:
            page.insert_text(point, text, fontsize=fontsize, fontname=fontname)
    path = tmp_path / "test.pdf"
    doc.save(path)
    doc.close()
    return path


def test_extracts_text_page_and_bold(tmp_path: Path) -> None:
    path = _make_pdf(tmp_path, [[((50, 50), "Plain body text", 10, "helv")]])

    parsed = parse_pdf(path)

    assert len(parsed.blocks) == 1
    block = parsed.blocks[0]
    assert block.text == "Plain body text"
    assert block.page == 1
    assert block.kind == "text"
    assert block.font_size == 10
    assert block.bold is False


def test_detects_title_by_font_size_above_median(tmp_path: Path) -> None:
    path = _make_pdf(
        tmp_path,
        [
            [
                ((50, 50), "Allgemeine Vorschriften", 14, "hebo"),
                ((50, 100), "Normal body sentence.", 10, "helv"),
            ]
        ],
    )

    parsed = parse_pdf(path)

    titles = {b.text: b.is_title for b in parsed.blocks}
    assert titles["Allgemeine Vorschriften"] is True
    assert titles["Normal body sentence."] is False


def test_detects_numbered_heading_regardless_of_font_size(tmp_path: Path) -> None:
    path = _make_pdf(
        tmp_path,
        [
            [
                ((50, 50), "1.1 Unterabschnitt", 10, "helv"),
                ((50, 100), "Normal body sentence.", 10, "helv"),
            ]
        ],
    )

    parsed = parse_pdf(path)

    titles = {b.text: b.is_title for b in parsed.blocks}
    assert titles["1.1 Unterabschnitt"] is True
    assert titles["Normal body sentence."] is False


def test_detects_paragraph_heading(tmp_path: Path) -> None:
    path = _make_pdf(tmp_path, [[((50, 50), "§ 5 Sonstiges", 10, "helv")]])

    parsed = parse_pdf(path)

    assert parsed.blocks[0].is_title is True


def test_section_path_is_inherited_by_following_blocks(tmp_path: Path) -> None:
    path = _make_pdf(
        tmp_path,
        [
            [
                ((50, 50), "1 Allgemeines", 10, "helv"),
                ((50, 100), "Body under section 1.", 10, "helv"),
                ((50, 150), "1.1 Unterabschnitt", 10, "helv"),
                ((50, 200), "Body under section 1.1.", 10, "helv"),
            ]
        ],
    )

    parsed = parse_pdf(path)

    by_text = {b.text: b.section_path for b in parsed.blocks}
    assert by_text["1 Allgemeines"] == "1"
    assert by_text["Body under section 1."] == "1"
    assert by_text["1.1 Unterabschnitt"] == "1.1"
    assert by_text["Body under section 1.1."] == "1.1"


def test_filters_repeated_header_across_pages(tmp_path: Path) -> None:
    pages = [
        [
            ((50, 20), "Running header", 8, "helv"),
            ((50, 100 + i * 20), f"Unique body {i}", 10, "helv"),
        ]
        for i in range(3)
    ]
    path = _make_pdf(tmp_path, pages)

    parsed = parse_pdf(path)

    texts = [b.text for b in parsed.blocks]
    assert "Running header" not in texts
    assert texts == ["Unique body 0", "Unique body 1", "Unique body 2"]


def test_keeps_non_repeated_text_on_single_page(tmp_path: Path) -> None:
    path = _make_pdf(tmp_path, [[((50, 50), "Only one page here.", 10, "helv")]])

    parsed = parse_pdf(path)

    assert [b.text for b in parsed.blocks] == ["Only one page here."]


def test_table_to_markdown_handles_none_and_multiline_cells() -> None:
    rows: list[list[str | None]] = [
        ["Untersuchungspunkt", "Kriterium", None],
        ["Bremse", "Zustand\nAuffälligkeiten", "Prüfzeichen"],
    ]

    markdown = _table_to_markdown(rows)

    lines = markdown.splitlines()
    assert lines[0] == "| Untersuchungspunkt | Kriterium |  |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| Bremse | Zustand / Auffälligkeiten | Prüfzeichen |"


def test_table_to_markdown_empty_rows_returns_empty_string() -> None:
    assert _table_to_markdown([]) == ""
