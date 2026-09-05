"""Pure unit tests for domain/conflicts.py.

Conflicts are grouped by requirement code, not by document_id: one
Document row has exactly one version_label, so every chunk under one
document_id already shares it -- a document_id grouping could never
fire. See the module docstring for why.
"""

import uuid

from app.domain.conflicts import find_version_conflicts
from app.domain.context import Source


def _source(
    marker: str,
    *,
    content: str = "Belanglos.",
    version_label: str | None = None,
) -> Source:
    return Source(
        marker=marker,
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="Lastenheft.docx",
        page_from=1,
        page_to=1,
        section_path=None,
        heading=None,
        content=content,
        version_label=version_label,
    )


def test_the_same_code_with_two_different_labels_is_a_conflict() -> None:
    sources = {
        "S1": _source("S1", content="LH-3.2.1 fordert 300 N.", version_label="v1.2"),
        "S2": _source("S2", content="LH-3.2.1 fordert 250 N.", version_label="v2.0"),
    }

    assert find_version_conflicts(sources) == {"lh3.2.1": ["v1.2", "v2.0"]}


def test_the_same_code_with_the_same_label_twice_is_not_a_conflict() -> None:
    sources = {
        "S1": _source("S1", content="LH-3.2.1 fordert 300 N.", version_label="v1.2"),
        "S2": _source("S2", content="Siehe auch LH-3.2.1.", version_label="v1.2"),
    }

    assert find_version_conflicts(sources) == {}


def test_a_single_source_is_never_a_conflict() -> None:
    sources = {"S1": _source("S1", content="LH-3.2.1 fordert 300 N.", version_label="v1.2")}

    assert find_version_conflicts(sources) == {}


def test_no_version_label_never_conflicts() -> None:
    sources = {
        "S1": _source("S1", content="LH-3.2.1 fordert 300 N.", version_label=None),
        "S2": _source("S2", content="LH-3.2.1 fordert 250 N.", version_label=None),
    }

    assert find_version_conflicts(sources) == {}


def test_a_labelled_and_an_unlabelled_source_do_not_conflict() -> None:
    sources = {
        "S1": _source("S1", content="LH-3.2.1 fordert 300 N.", version_label="v1.2"),
        "S2": _source("S2", content="LH-3.2.1 fordert 250 N.", version_label=None),
    }

    assert find_version_conflicts(sources) == {}


def test_different_codes_with_different_labels_do_not_conflict() -> None:
    sources = {
        "S1": _source("S1", content="LH-3.2.1 fordert 300 N.", version_label="v1.2"),
        "S2": _source("S2", content="LH-4.1.0 regelt etwas anderes.", version_label="v2.0"),
    }

    assert find_version_conflicts(sources) == {}


def test_a_source_with_no_code_at_all_never_conflicts() -> None:
    sources = {
        "S1": _source("S1", content="Allgemeiner Fliesstext ohne Code.", version_label="v1.2"),
        "S2": _source("S2", content="Noch mehr Fliesstext.", version_label="v2.0"),
    }

    assert find_version_conflicts(sources) == {}


def test_labels_are_ordered_by_first_appearance_and_deduplicated() -> None:
    sources = {
        "S1": _source("S1", content="LH-3.2.1: 250 N.", version_label="v2.0"),
        "S2": _source("S2", content="LH-3.2.1: 300 N.", version_label="v1.2"),
        "S3": _source("S3", content="LH-3.2.1 erneut.", version_label="v2.0"),
    }

    assert find_version_conflicts(sources) == {"lh3.2.1": ["v2.0", "v1.2"]}


def test_empty_sources_is_never_a_conflict() -> None:
    assert find_version_conflicts({}) == {}
