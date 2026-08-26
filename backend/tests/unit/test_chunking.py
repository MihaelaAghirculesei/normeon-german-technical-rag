import dataclasses

from app.domain.chunking import ALL_STRATEGIES, Chunk, FixedWindowChunker, StructuralChunker
from app.domain.parsing import Block, ParsedDocument


def _block(
    text: str,
    page: int = 1,
    section_path: str | None = None,
    is_title: bool = False,
    kind: str = "text",
) -> Block:
    return Block(
        text=text,
        page=page,
        bbox=(0.0, 0.0, 100.0, 10.0),
        font_size=10.0,
        bold=is_title,
        kind=kind,  # type: ignore[arg-type]
        is_title=is_title,
        section_path=section_path,
    )


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


# --- FixedWindowChunker ------------------------------------------------


def test_fixed_empty_document_returns_no_chunks() -> None:
    assert FixedWindowChunker().chunk(ParsedDocument(blocks=[])) == []


def test_fixed_short_document_is_a_single_chunk() -> None:
    doc = ParsedDocument(blocks=[_block(_words(50), page=1)])

    chunks = FixedWindowChunker().chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].token_count == 50
    assert chunks[0].page_from == 1
    assert chunks[0].page_to == 1


def test_fixed_window_size_and_step_are_configurable() -> None:
    chunker = FixedWindowChunker(window_tokens=100, overlap_ratio=0.5)
    assert chunker.window_tokens == 100
    assert chunker.step == 50


def test_fixed_exact_multiple_of_window_has_no_tiny_trailing_chunk() -> None:
    chunker = FixedWindowChunker(window_tokens=100, overlap_ratio=0.0)
    doc = ParsedDocument(blocks=[_block(_words(200))])

    chunks = chunker.chunk(doc)

    assert len(chunks) == 2
    assert chunks[0].token_count == 100
    assert chunks[1].token_count == 100


def test_fixed_windows_overlap_by_the_configured_ratio() -> None:
    chunker = FixedWindowChunker(window_tokens=100, overlap_ratio=0.15)
    doc = ParsedDocument(blocks=[_block(_words(1000))])

    chunks = chunker.chunk(doc)

    first_words = set(chunks[0].content.split())
    second_words = set(chunks[1].content.split())
    assert len(first_words & second_words) == 15


def test_fixed_last_window_is_shorter_when_document_does_not_divide_evenly() -> None:
    chunker = FixedWindowChunker(window_tokens=100, overlap_ratio=0.0)
    doc = ParsedDocument(blocks=[_block(_words(150))])

    chunks = chunker.chunk(doc)

    assert len(chunks) == 2
    assert chunks[0].token_count == 100
    assert chunks[1].token_count == 50


def test_fixed_ordinals_increase_sequentially() -> None:
    chunker = FixedWindowChunker(window_tokens=50, overlap_ratio=0.0)
    doc = ParsedDocument(blocks=[_block(_words(150))])

    chunks = chunker.chunk(doc)

    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_fixed_page_span_crosses_a_page_boundary() -> None:
    chunker = FixedWindowChunker(window_tokens=100, overlap_ratio=0.0)
    doc = ParsedDocument(
        blocks=[_block(_words(60), page=1), _block(_words(60), page=2)]
    )

    chunks = chunker.chunk(doc)

    assert chunks[0].page_from == 1
    assert chunks[0].page_to == 2


def test_fixed_section_path_is_taken_from_the_windows_first_word() -> None:
    chunker = FixedWindowChunker(window_tokens=100, overlap_ratio=0.0)
    doc = ParsedDocument(
        blocks=[
            _block(_words(30), section_path="1"),
            _block(_words(30), section_path="2"),
        ]
    )

    chunks = chunker.chunk(doc)

    assert chunks[0].section_path == "1"


def test_fixed_heading_is_always_none() -> None:
    doc = ParsedDocument(blocks=[_block(_words(10))])
    assert FixedWindowChunker().chunk(doc)[0].heading is None


def test_fixed_parent_ordinal_is_always_none() -> None:
    doc = ParsedDocument(blocks=[_block(_words(10))])
    assert FixedWindowChunker().chunk(doc)[0].parent_ordinal is None


def test_fixed_token_count_matches_actual_content_word_count() -> None:
    chunker = FixedWindowChunker(window_tokens=37, overlap_ratio=0.0)
    doc = ParsedDocument(blocks=[_block(_words(37))])

    chunk = chunker.chunk(doc)[0]

    assert chunk.token_count == len(chunk.content.split())


# --- StructuralChunker --------------------------------------------------


def test_structural_empty_document_returns_no_chunks() -> None:
    assert StructuralChunker().chunk(ParsedDocument(blocks=[])) == []


def test_structural_single_section_becomes_one_chunk() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Allgemeines", section_path="1", is_title=True),
            _block(_words(50), section_path="1"),
        ]
    )

    chunks = StructuralChunker().chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].section_path == "1"


def test_structural_content_before_first_heading_has_no_section_path() -> None:
    doc = ParsedDocument(blocks=[_block(_words(150))])

    chunks = StructuralChunker().chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].section_path is None
    assert chunks[0].heading is None


def test_structural_breadcrumb_chains_nested_numbered_sections() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Allgemeines", section_path="1", is_title=True),
            _block(_words(150), section_path="1"),
            _block("1.1 Unterabschnitt", section_path="1.1", is_title=True),
            _block(_words(150), section_path="1.1"),
        ]
    )

    chunks = StructuralChunker().chunk(doc)
    leaf = next(c for c in chunks if c.section_path == "1.1")

    assert leaf.heading == "1 Allgemeines > 1.1 Unterabschnitt"


def test_structural_flat_paragraph_heading_has_no_ancestor_chain() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("§19 Erteilung der Betriebserlaubnis", section_path="§19", is_title=True),
            _block(_words(150), section_path="§19"),
        ]
    )

    chunks = StructuralChunker().chunk(doc)

    assert chunks[0].heading == "§19 Erteilung der Betriebserlaubnis"


def test_structural_breadcrumb_is_prepended_to_chunk_content() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Allgemeines", section_path="1", is_title=True),
            _block(_words(150), section_path="1"),
        ]
    )

    chunk = StructuralChunker().chunk(doc)[0]

    assert chunk.content.startswith("1 Allgemeines\n\n")


def test_structural_small_section_merges_into_next_sibling() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Tiny", section_path="1", is_title=True),
            _block(_words(10), section_path="1"),
            _block("2 Bigger", section_path="2", is_title=True),
            _block(_words(150), section_path="2"),
        ]
    )

    chunks = StructuralChunker().chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].section_path == "2"
    assert "1 Tiny" in chunks[0].content


def test_structural_cascading_merge_across_two_tiny_sections() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Tiny", section_path="1", is_title=True),
            _block(_words(5), section_path="1"),
            _block("2 AlsoTiny", section_path="2", is_title=True),
            _block(_words(5), section_path="2"),
            _block("3 Bigger", section_path="3", is_title=True),
            _block(_words(150), section_path="3"),
        ]
    )

    chunks = StructuralChunker().chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].section_path == "3"
    assert "1 Tiny" in chunks[0].content
    assert "2 AlsoTiny" in chunks[0].content


def test_structural_trailing_tiny_section_with_no_sibling_is_kept_alone() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Bigger", section_path="1", is_title=True),
            _block(_words(150), section_path="1"),
            _block("2 TrailingTiny", section_path="2", is_title=True),
            _block(_words(5), section_path="2"),
        ]
    )

    chunks = StructuralChunker().chunk(doc)

    assert len(chunks) == 2
    assert chunks[1].section_path == "2"


def test_structural_large_section_is_split_into_multiple_chunks() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Big", section_path="1", is_title=True),
            _block(_words(1000), section_path="1"),
        ]
    )

    chunks = StructuralChunker().chunk(doc)

    assert len(chunks) == 2
    assert all(c.section_path == "1" for c in chunks)


def test_structural_split_pieces_respect_the_max_token_budget() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Big", section_path="1", is_title=True),  # 2 words, counted too
            _block(_words(998), section_path="1"),
        ]
    )

    chunks = StructuralChunker().chunk(doc)
    body_lengths = [len(c.content.split("\n\n", 1)[1].split()) for c in chunks]

    assert body_lengths[0] == 800
    assert body_lengths[1] == 200


def test_structural_first_split_piece_has_no_parent_ordinal() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Big", section_path="1", is_title=True),
            _block(_words(1000), section_path="1"),
        ]
    )

    chunks = StructuralChunker().chunk(doc)

    assert chunks[0].parent_ordinal is None


def test_structural_later_split_pieces_point_back_to_the_first() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Big", section_path="1", is_title=True),
            _block(_words(1000), section_path="1"),
        ]
    )

    chunks = StructuralChunker().chunk(doc)

    assert chunks[1].parent_ordinal == chunks[0].ordinal


def test_structural_split_pieces_carry_the_pages_of_their_own_words_only() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Big", section_path="1", is_title=True, page=1),  # 2 words
            _block(_words(798), section_path="1", page=1),
            _block(_words(200), section_path="1", page=2),
        ]
    )

    chunks = StructuralChunker().chunk(doc)

    assert chunks[0].page_from == 1
    assert chunks[0].page_to == 1
    assert chunks[1].page_from == 2
    assert chunks[1].page_to == 2


def test_structural_ordinals_are_sequential_across_the_whole_document() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 First", section_path="1", is_title=True),
            _block(_words(150), section_path="1"),
            _block("2 Second", section_path="2", is_title=True),
            _block(_words(150), section_path="2"),
        ]
    )

    chunks = StructuralChunker().chunk(doc)

    assert [c.ordinal for c in chunks] == [0, 1]


def test_structural_table_blocks_are_included_in_section_body() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 WithTable", section_path="1", is_title=True),
            _block(_words(50), section_path="1"),
            _block("| a | b |\n| --- | --- |\n| 1 | 2 |", section_path="1", kind="table"),
        ]
    )

    chunk = StructuralChunker().chunk(doc)[0]

    assert "| a | b |" in chunk.content


def test_structural_heading_field_matches_the_prepended_breadcrumb() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Allgemeines", section_path="1", is_title=True),
            _block(_words(150), section_path="1"),
        ]
    )

    chunk = StructuralChunker().chunk(doc)[0]

    assert chunk.content.startswith(chunk.heading or "")


def test_structural_consecutive_blocks_in_the_same_section_form_one_chunk() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Allgemeines", section_path="1", is_title=True),
            _block(_words(30), section_path="1"),
            _block(_words(30), section_path="1"),
            _block(_words(30), section_path="1"),
        ]
    )

    chunks = StructuralChunker().chunk(doc)

    assert len(chunks) == 1


# --- Cross-strategy parity ------------------------------------------------


def test_all_strategies_are_registered() -> None:
    names = {s.name for s in ALL_STRATEGIES}
    assert names == {"fixed_500", "structural"}


def test_both_strategies_produce_the_same_chunk_schema() -> None:
    doc = ParsedDocument(
        blocks=[
            _block("1 Allgemeines", section_path="1", is_title=True),
            _block(_words(600), section_path="1"),
        ]
    )

    fixed_chunks = FixedWindowChunker().chunk(doc)
    structural_chunks = StructuralChunker().chunk(doc)

    fixed_fields = {f.name for f in dataclasses.fields(fixed_chunks[0])}
    structural_fields = {f.name for f in dataclasses.fields(structural_chunks[0])}
    assert fixed_fields == structural_fields
    assert isinstance(fixed_chunks[0], Chunk)
    assert isinstance(structural_chunks[0], Chunk)
