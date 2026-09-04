"""Unit tests for domain/citations.py -- pure, no DB/model involved.

Covers the Day 12 "Fatto quando": an answer with an invented marker has
it removed from the text and reported separately.
"""

import uuid

from app.domain.citations import Citation, extract_and_validate
from app.domain.context import Source


def _source(marker: str, content: str = "Der Regelungstext.") -> Source:
    return Source(
        marker=marker,
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="StVZO.pdf",
        page_from=14,
        page_to=14,
        section_path="50",
        heading="Scheinwerfer",
        content=content,
    )


def test_all_markers_valid_are_returned_as_citations_unchanged() -> None:
    sources = {"S1": _source("S1"), "S2": _source("S2")}

    answer, citations, invented = extract_and_validate(
        "Es gilt Regel A [S1] sowie Regel B [S2].", sources
    )

    assert answer == "Es gilt Regel A [S1] sowie Regel B [S2]."
    assert [c.marker for c in citations] == ["S1", "S2"]
    assert invented == []


def test_an_invented_marker_is_removed_from_the_answer_text() -> None:
    """The literal Day 12 acceptance check: an invented [S7] disappears."""
    sources = {"S1": _source("S1")}

    answer, citations, invented = extract_and_validate(
        "Es gilt Regel A [S1] und angeblich Regel G [S7].", sources
    )

    assert "[S7]" not in answer
    assert invented == ["S7"]
    assert [c.marker for c in citations] == ["S1"]


def test_removing_a_marker_does_not_leave_a_double_space() -> None:
    sources = {"S1": _source("S1")}

    answer, _, _ = extract_and_validate("Regel A [S1] und Regel G [S7] gelten.", sources)

    assert "  " not in answer
    assert answer == "Regel A [S1] und Regel G gelten."


def test_zero_valid_citations_with_claims_is_treated_as_abstention() -> None:
    sources = {"S1": _source("S1")}

    answer, citations, invented = extract_and_validate(
        "Angeblich gilt Regel G [S7].", sources
    )

    assert answer == "NICHT_GEFUNDEN"
    assert citations == []
    assert invented == ["S7"]


def test_claims_with_no_marker_at_all_is_also_an_abstention() -> None:
    sources = {"S1": _source("S1")}

    answer, citations, invented = extract_and_validate(
        "Es gelten diverse Anforderungen.", sources
    )

    assert answer == "NICHT_GEFUNDEN"
    assert citations == []
    assert invented == []


def test_nicht_gefunden_passes_through_unchanged() -> None:
    sources = {"S1": _source("S1")}

    answer, citations, invented = extract_and_validate("NICHT_GEFUNDEN", sources)

    assert answer == "NICHT_GEFUNDEN"
    assert citations == []
    assert invented == []


def test_repeated_markers_produce_one_citation() -> None:
    sources = {"S1": _source("S1")}

    _, citations, _ = extract_and_validate("[S1] Regel A. Siehe auch [S1].", sources)

    assert len(citations) == 1
    assert citations[0].marker == "S1"


def test_citations_are_ordered_by_first_appearance_not_by_marker_number() -> None:
    sources = {"S1": _source("S1"), "S2": _source("S2")}

    _, citations, _ = extract_and_validate("Zuerst [S2], dann [S1].", sources)

    assert [c.marker for c in citations] == ["S2", "S1"]


def test_citation_snippet_is_the_stripped_content_when_short() -> None:
    sources = {"S1": _source("S1", content="  Kurzer Text.  ")}

    _, citations, _ = extract_and_validate("[S1]", sources)

    assert citations[0].snippet == "Kurzer Text."


def test_citation_snippet_is_truncated_when_long() -> None:
    sources = {"S1": _source("S1", content="x" * 500)}

    _, citations, _ = extract_and_validate("[S1]", sources)

    assert len(citations[0].snippet) == 241  # 240 chars + ellipsis
    assert citations[0].snippet.endswith("…")


def test_citation_carries_the_source_metadata() -> None:
    source = _source("S1")
    sources = {"S1": source}

    _, citations, _ = extract_and_validate("[S1]", sources)

    citation = citations[0]
    assert isinstance(citation, Citation)
    assert citation.chunk_id == source.chunk_id
    assert citation.document_id == source.document_id
    assert citation.filename == source.filename
    assert citation.page_from == source.page_from
    assert citation.page_to == source.page_to
    assert citation.section_path == source.section_path


def test_an_all_invented_answer_abstains_even_after_stripping_leaves_prose() -> None:
    sources = {"S1": _source("S1")}

    answer, citations, invented = extract_and_validate(
        "Text A [S7] und Text B [S8].", sources
    )

    assert answer == "NICHT_GEFUNDEN"
    assert citations == []
    assert invented == ["S7", "S8"]
