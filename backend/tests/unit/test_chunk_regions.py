"""Chunks remember where their blocks were, for the PDF viewer."""

from app.domain.chunking import FixedWindowChunker, Region, StructuralChunker
from app.domain.parsing import Block, ParsedDocument

BBOX_1 = (10.0, 20.0, 300.0, 40.0)
BBOX_2 = (10.0, 50.0, 300.0, 70.0)


def _block(text: str, page: int, bbox: tuple[float, float, float, float]) -> Block:
    return Block(text=text, page=page, bbox=bbox, font_size=10.0, bold=False)


def _document() -> ParsedDocument:
    return ParsedDocument(
        blocks=[_block("erster Absatz", 1, BBOX_1), _block("zweiter Absatz", 2, BBOX_2)]
    )


def test_a_fixed_window_chunk_lists_each_block_once_in_reading_order() -> None:
    chunks = FixedWindowChunker(window_tokens=10).chunk(_document())

    assert chunks[0].regions == (Region(1, BBOX_1), Region(2, BBOX_2))


def test_a_fixed_window_chunk_lists_only_the_blocks_it_covers() -> None:
    chunks = FixedWindowChunker(window_tokens=2, overlap_ratio=0).chunk(_document())

    assert [c.regions for c in chunks] == [(Region(1, BBOX_1),), (Region(2, BBOX_2),)]


def test_a_structural_chunk_lists_the_blocks_of_its_section() -> None:
    chunks = StructuralChunker().chunk(_document())

    assert chunks[0].regions == (Region(1, BBOX_1), Region(2, BBOX_2))


def test_regions_do_not_change_the_content_or_its_hash_input() -> None:
    chunks = FixedWindowChunker(window_tokens=10).chunk(_document())

    assert chunks[0].content == "erster Absatz zweiter Absatz"
