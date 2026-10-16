"""Turn retrieved chunks into the exact text block the model sees, plus a
stable ``marker -> Source`` map the backend keeps to itself.

The model only ever cites ``[S1]``, ``[S2]``, ...; it is never shown or
asked for page numbers or filenames. This module owns the mapping from
those markers back to the real citation metadata -- that split is what
makes the citations verifiable (plan, Giorno 11, "NON fare").

Pure: no I/O, no settings, no logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from app.domain.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class Source:
    """One numbered source sitting behind a ``[S{n}]`` marker."""

    marker: str  # "S1", "S2", ...
    chunk_id: UUID
    document_id: UUID
    filename: str
    page_from: int
    page_to: int
    section_path: str | None
    heading: str | None
    content: str
    version_label: str | None = None
    ocr: bool = False


def _seiten(page_from: int, page_to: int) -> str:
    if page_to > page_from:
        return f"Seite {page_from}–{page_to}"
    return f"Seite {page_from}"


def _header(source: Source) -> str:
    parts = [f"Quelle: {source.filename}"]
    if source.version_label:
        parts.append(f"Version {source.version_label}")
    parts.append(_seiten(source.page_from, source.page_to))
    if source.section_path:
        parts.append(f"Abschnitt {source.section_path}")
    return " | ".join(parts)


_TAG = re.compile(r"<\s*(/?)\s*(quellen?)\b", re.IGNORECASE)
_FORGED_MARKER = re.compile(r"\[(S\d+)\]")


def _neutralise(text: str) -> str:
    """Make document text unable to act as prompt structure: a ``</quelle>``
    inside a document cannot close its own delimiter early (and so cannot
    smuggle text outside it), and an ``[S3]`` inside a document cannot pose
    as a citation marker of another source. Both are rewritten into
    look-alikes the model still reads as text."""
    text = _TAG.sub(lambda m: f"‹{m.group(1)}{m.group(2)}", text)
    return _FORGED_MARKER.sub(r"(\1)", text)


def _attribute(value: str) -> str:
    value = value.replace('"', "'").replace("<", "‹").replace(">", "›")
    return _FORGED_MARKER.sub(r"(\1)", value)


def _untrusted_block(source: Source) -> str:
    """One source as delimited, untrusted data. Deliberately without the
    filename and pages: the model cites markers only, and what it never
    sees it cannot be talked into revealing (Giorno 26)."""
    attrs = [f'marker="[{source.marker}]"']
    if source.version_label:
        attrs.append(f'version="{_attribute(source.version_label)}"')
    if source.section_path:
        attrs.append(f'abschnitt="{_attribute(source.section_path)}"')
    body = _neutralise(source.content.strip())
    return f"<quelle {' '.join(attrs)}>\n{body}\n</quelle>"


def build_context(
    chunks: list[RetrievedChunk], *, untrusted: bool = False
) -> tuple[str, dict[str, Source]]:
    """Assign ``S1..Sn`` to ``chunks`` in order and render the context block.

    The v1 form (``untrusted=False``), kept so earlier evaluation runs stay
    reproducible::

        [S1] Quelle: StVZO.pdf | Seite 14 | Abschnitt 5.1.2
        <chunk text>

    The untrusted-data form (``untrusted=True``, prompt v2 onwards), which
    a prompt-injection paragraph in a document cannot break out of::

        <quellen>
        <quelle marker="[S1]" abschnitt="5.1.2">
        <chunk text, neutralised>
        </quelle>
        </quellen>

    Returns ``(block, {marker: Source})``. An empty input gives an empty
    string and an empty map; the caller decides what that means (Day 13
    turns it into a pre-generation abstention).
    """
    sources: dict[str, Source] = {}
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        marker = f"S{i}"
        source = Source(
            marker=marker,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_from=chunk.page_from,
            page_to=chunk.page_to,
            section_path=chunk.section_path,
            heading=chunk.heading,
            content=chunk.content,
            version_label=chunk.version_label,
            ocr=chunk.ocr,
        )
        sources[marker] = source
        if untrusted:
            blocks.append(_untrusted_block(source))
        else:
            blocks.append(f"[{marker}] {_header(source)}\n{chunk.content.strip()}")
    if untrusted and blocks:
        return "<quellen>\n" + "\n\n".join(blocks) + "\n</quellen>", sources
    return "\n\n".join(blocks), sources
