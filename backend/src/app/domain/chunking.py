from dataclasses import dataclass, field
from typing import Protocol

from app.domain.parsing import Block, ParsedDocument

_MIN_SECTION_TOKENS = 100
_MAX_SECTION_TOKENS = 800
_FIXED_WINDOW_TOKENS = 500
_FIXED_WINDOW_OVERLAP = 0.15


def _estimate_tokens(text: str) -> int:
    """Whitespace-word count, used as a token proxy until Day 5 wires in the
    real embedding model's tokenizer. Stable and cheap to test; the ~500/800/
    100 thresholds in the plan are approximate anyway."""
    return len(text.split())


@dataclass(frozen=True)
class Chunk:
    content: str
    page_from: int
    page_to: int
    section_path: str | None
    heading: str | None
    ordinal: int
    token_count: int
    parent_ordinal: int | None = None


class ChunkingStrategy(Protocol):
    name: str

    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...


@dataclass(frozen=True)
class _Word:
    text: str
    page: int
    section_path: str | None = None


def _flatten_words(blocks: list[Block]) -> list[_Word]:
    return [
        _Word(token, block.page, block.section_path)
        for block in blocks
        for token in block.text.split()
    ]


class FixedWindowChunker:
    """Strategy A: a fixed token window over the whole document's
    concatenated text, with overlap -- ignores section boundaries."""

    name = "fixed_500"

    def __init__(
        self,
        window_tokens: int = _FIXED_WINDOW_TOKENS,
        overlap_ratio: float = _FIXED_WINDOW_OVERLAP,
    ) -> None:
        self.window_tokens = window_tokens
        self.step = max(1, round(window_tokens * (1 - overlap_ratio)))

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        words = _flatten_words(doc.blocks)
        if not words:
            return []

        chunks: list[Chunk] = []
        total = len(words)
        start = 0
        ordinal = 0
        while start < total:
            end = min(start + self.window_tokens, total)
            window = words[start:end]
            chunks.append(
                Chunk(
                    content=" ".join(w.text for w in window),
                    page_from=min(w.page for w in window),
                    page_to=max(w.page for w in window),
                    section_path=window[0].section_path,
                    heading=None,
                    ordinal=ordinal,
                    token_count=len(window),
                )
            )
            ordinal += 1
            if end == total:
                break
            start += self.step
        return chunks


@dataclass
class _LeafSection:
    section_path: str | None
    blocks: list[Block] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return _estimate_tokens("\n\n".join(b.text for b in self.blocks))


def _group_into_leaf_sections(blocks: list[Block]) -> list[_LeafSection]:
    groups: list[_LeafSection] = []
    for block in blocks:
        if groups and groups[-1].section_path == block.section_path:
            groups[-1].blocks.append(block)
        else:
            groups.append(_LeafSection(section_path=block.section_path, blocks=[block]))
    return groups


def _merge_small_sections(groups: list[_LeafSection]) -> list[_LeafSection]:
    """A section under the token floor merges into its next sibling rather
    than becoming a near-useless standalone chunk; the merged chunk takes on
    the *next* sibling's identity since that's the section that survives."""
    merged: list[_LeafSection] = []
    pending: _LeafSection | None = None
    for group in groups:
        current = group
        if pending is not None:
            current = _LeafSection(
                section_path=group.section_path, blocks=pending.blocks + group.blocks
            )
            pending = None
        if current.token_count < _MIN_SECTION_TOKENS:
            pending = current
        else:
            merged.append(current)
    if pending is not None:
        merged.append(pending)
    return merged


def _heading_by_path(blocks: list[Block]) -> dict[str, str]:
    heading_by_path: dict[str, str] = {}
    for block in blocks:
        if block.is_title and block.section_path and block.section_path not in heading_by_path:
            heading_by_path[block.section_path] = block.text
    return heading_by_path


def _breadcrumb(section_path: str | None, heading_by_path: dict[str, str]) -> str | None:
    if section_path is None:
        return None
    parts = section_path.split(".")
    ancestor_codes = [".".join(parts[: i + 1]) for i in range(len(parts))]
    labels = [heading_by_path[code] for code in ancestor_codes if code in heading_by_path]
    return " > ".join(labels) if labels else None


class StructuralChunker:
    """Strategy B: one chunk per leaf section, tiny sections folded into
    their next sibling, oversized sections split with parent_ordinal linking
    the pieces back to the first one. Every chunk carries its breadcrumb at
    the top of `content` (helps both embedding and citation readability)."""

    name = "structural"

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        heading_by_path = _heading_by_path(doc.blocks)
        groups = _merge_small_sections(_group_into_leaf_sections(doc.blocks))

        chunks: list[Chunk] = []
        ordinal = 0
        for group in groups:
            words = [(token, b.page) for b in group.blocks for token in b.text.split()]
            if not words:
                continue

            breadcrumb = _breadcrumb(group.section_path, heading_by_path)
            pieces = [
                words[i : i + _MAX_SECTION_TOKENS]
                for i in range(0, len(words), _MAX_SECTION_TOKENS)
            ]

            parent_ordinal: int | None = None
            for i, piece in enumerate(pieces):
                body = " ".join(token for token, _ in piece)
                content = f"{breadcrumb}\n\n{body}" if breadcrumb else body
                this_ordinal = ordinal
                chunks.append(
                    Chunk(
                        content=content,
                        page_from=min(p for _, p in piece),
                        page_to=max(p for _, p in piece),
                        section_path=group.section_path,
                        heading=breadcrumb,
                        ordinal=this_ordinal,
                        token_count=_estimate_tokens(content),
                        parent_ordinal=parent_ordinal,
                    )
                )
                if i == 0:
                    parent_ordinal = this_ordinal
                ordinal += 1
        return chunks


ALL_STRATEGIES: list[ChunkingStrategy] = [FixedWindowChunker(), StructuralChunker()]
