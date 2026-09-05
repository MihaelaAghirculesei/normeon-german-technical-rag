"""Pure unit tests for ``build_context`` -- marker assignment, the block
format the model sees, and the marker -> citation-metadata map the
backend keeps.
"""

import uuid

from app.domain.context import Source, build_context
from app.domain.models import RetrievedChunk


def _chunk(
    *,
    content: str = "Die zulaessige Lenkkraft betraegt 300 N.",
    page_from: int = 14,
    page_to: int = 14,
    section_path: str | None = "5.1.2",
    filename: str = "UN-R79.pdf",
    version_label: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=filename,
        content=content,
        page_from=page_from,
        page_to=page_to,
        section_path=section_path,
        heading="Lenkkraft",
        score=0.9,
        version_label=version_label,
    )


def test_markers_are_assigned_s1_upwards_in_order() -> None:
    a, b, c = _chunk(), _chunk(), _chunk()

    block, sources = build_context([a, b, c])

    assert list(sources) == ["S1", "S2", "S3"]
    assert sources["S1"].chunk_id == a.chunk_id
    assert sources["S3"].chunk_id == c.chunk_id
    assert block.index("[S1]") < block.index("[S2]") < block.index("[S3]")


def test_block_format_has_the_marked_header_then_the_text() -> None:
    block, _ = build_context([_chunk()])

    assert block == (
        "[S1] Quelle: UN-R79.pdf | Seite 14 | Abschnitt 5.1.2\n"
        "Die zulaessige Lenkkraft betraegt 300 N."
    )


def test_page_range_is_rendered_when_the_chunk_spans_pages() -> None:
    block, _ = build_context([_chunk(page_from=14, page_to=16)])

    assert "Seite 14–16" in block


def test_missing_section_path_drops_the_abschnitt_part() -> None:
    block, _ = build_context([_chunk(section_path=None)])

    assert "Abschnitt" not in block
    assert "[S1] Quelle: UN-R79.pdf | Seite 14\n" in block


def test_blocks_are_separated_by_a_blank_line_and_content_is_stripped() -> None:
    block, _ = build_context([_chunk(content="  vorne  "), _chunk(content="hinten")])

    assert block == (
        "[S1] Quelle: UN-R79.pdf | Seite 14 | Abschnitt 5.1.2\nvorne\n\n"
        "[S2] Quelle: UN-R79.pdf | Seite 14 | Abschnitt 5.1.2\nhinten"
    )


def test_empty_input_gives_an_empty_block_and_map() -> None:
    assert build_context([]) == ("", {})


def test_source_carries_the_citation_metadata_from_the_chunk() -> None:
    chunk = _chunk(filename="FZV.pdf", page_from=3, page_to=3)

    _, sources = build_context([chunk])

    src = sources["S1"]
    assert isinstance(src, Source)
    assert (src.document_id, src.filename, src.page_from) == (
        chunk.document_id,
        "FZV.pdf",
        3,
    )


def test_braces_in_chunk_text_are_passed_through_verbatim() -> None:
    block, _ = build_context([_chunk(content="Grenzwert a{b} bei 50 % Last")])

    assert "Grenzwert a{b} bei 50 % Last" in block


def test_version_label_is_rendered_right_after_the_filename() -> None:
    block, sources = build_context([_chunk(version_label="1.2")])

    assert "[S1] Quelle: UN-R79.pdf | Version 1.2 | Seite 14 | Abschnitt 5.1.2\n" in block
    assert sources["S1"].version_label == "1.2"


def test_missing_version_label_is_omitted_from_the_header() -> None:
    block, sources = build_context([_chunk(version_label=None)])

    assert "Version" not in block
    assert sources["S1"].version_label is None
