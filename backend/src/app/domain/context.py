"""Turn retrieved chunks into the exact text block the model sees, plus a
stable ``marker -> Source`` map the backend keeps to itself.

The model only ever cites ``[S1]``, ``[S2]``, ...; it is never shown or
asked for page numbers or filenames. This module owns the mapping from
those markers back to the real citation metadata -- that split is what
makes the citations verifiable (plan, Giorno 11, "NON fare").

Pure: no I/O, no settings, no logging.
"""

from __future__ import annotations

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


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, dict[str, Source]]:
    """Assign ``S1..Sn`` to ``chunks`` in order and render the context block::

        [S1] Quelle: StVZO.pdf | Seite 14 | Abschnitt 5.1.2
        <chunk text>

        [S2] Quelle: FZV.pdf | Seite 3
        <chunk text>

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
        )
        sources[marker] = source
        blocks.append(f"[{marker}] {_header(source)}\n{chunk.content.strip()}")
    return "\n\n".join(blocks), sources
