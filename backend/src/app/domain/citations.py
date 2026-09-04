"""Validate the citation markers a model actually wrote against the
sources it was actually given (plan, Giorno 12).

A model can invent a marker that was never offered (`[S7]` when only
five sources exist) or, with a weak model/prompt, write claims with no
marker at all. Neither should ever surface to the user as if it were
grounded: `extract_and_validate` drops invented markers from the answer
text and -- if that leaves an answer with claims but zero real citations
-- treats the whole thing as an abstention.

Pure: no I/O, no settings, no logging, no metrics. Same split as
`domain/context.build_context` -- the caller (`services/generation.py`)
decides what an invented marker means for logs and metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from app.domain.context import Source

ABSTENTION_TEXT = "NICHT_GEFUNDEN"

_MARKER = re.compile(r"\[S\d+\]")
_SNIPPET_MAX_CHARS = 240


@dataclass(frozen=True, slots=True)
class Citation:
    """One `[S{n}]` marker the model wrote that resolved to a real
    source -- the shape the API exposes citations as."""

    marker: str
    chunk_id: UUID
    document_id: UUID
    filename: str
    page_from: int
    page_to: int
    section_path: str | None
    snippet: str


def _snippet(content: str) -> str:
    text = content.strip()
    if len(text) <= _SNIPPET_MAX_CHARS:
        return text
    return text[:_SNIPPET_MAX_CHARS].rstrip() + "…"


def _strip_marker(text: str, marker: str) -> str:
    """Remove one "[S7]" occurrence, taking a leading space with it so
    stripping doesn't leave a double space behind."""
    return re.sub(rf" ?\[{re.escape(marker)}\]", "", text)


def extract_and_validate(
    answer: str, sources: dict[str, Source]
) -> tuple[str, list[Citation], list[str]]:
    """Split the `[S..]` markers in `answer` into real citations (the
    marker is a key in `sources`) and invented ones, strip the invented
    markers from the text, and turn the result into an abstention if
    nothing real is left to cite.

    Returns `(answer, valid_citations, invented_markers)`:
      - `answer` -- the model's text with invented markers removed, or
        exactly `NICHT_GEFUNDEN` if it made claims (any non-abstention
        text) backed by zero valid citations.
      - `valid_citations` -- one `Citation` per distinct valid marker,
        in the order it first appears in `answer`.
      - `invented_markers` -- distinct markers that were not in
        `sources`, in first-appearance order. The caller logs these and
        bumps `hallucinated_citation_total`; this function never fails
        the request over them.
    """
    seen: list[str] = []
    for match in _MARKER.finditer(answer):
        marker = match.group()[1:-1]  # "[S1]" -> "S1"
        if marker not in seen:
            seen.append(marker)

    valid_markers = [m for m in seen if m in sources]
    invented_markers = [m for m in seen if m not in sources]

    cleaned = answer
    for marker in invented_markers:
        cleaned = _strip_marker(cleaned, marker)
    cleaned = cleaned.strip()

    citations = [
        Citation(
            marker=marker,
            chunk_id=sources[marker].chunk_id,
            document_id=sources[marker].document_id,
            filename=sources[marker].filename,
            page_from=sources[marker].page_from,
            page_to=sources[marker].page_to,
            section_path=sources[marker].section_path,
            snippet=_snippet(sources[marker].content),
        )
        for marker in valid_markers
    ]

    if cleaned != ABSTENTION_TEXT and not citations:
        return ABSTENTION_TEXT, [], invented_markers

    return cleaned, citations, invented_markers
