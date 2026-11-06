"""The README's results chart: recall@k by question category, one row per
configuration of the ablation matrix, as a static SVG heatmap.

Static on purpose: GitHub shows a README image without scripts, so every
value is printed in its cell, and the full tables stay in
docs/EVALUATION.md. One sequential hue, more is darker on the light theme
and brighter on the dark one; text picks its ink by contrast with the cell.
"""

from dataclasses import dataclass
from typing import Literal
from xml.sax.saxutils import escape

from app.eval.metrics import QuestionOutcome
from app.eval.models import CATEGORIES

Theme = Literal["light", "dark"]

# Categories with gold sources: an unanswerable question has nothing to recall.
RECALL_CATEGORIES = tuple(c for c in CATEGORIES if c != "unanswerable")
COLUMN_TITLES = {
    "all": "All answerable",
    "requirement_lookup": "Requirement lookup",
    "cross_reference": "Cross-reference",
    "code_lookup": "Code lookup",
    "conflict": "Version conflict",
}

# Blue sequential ramp, steps 100..650 (light to dark), without step 450:
# on it neither white nor near-black text reaches 4.5:1.
_RAMP = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#256abf", "#1c5cab", "#184f95", "#104281",
)  # fmt: skip
_THEMES: dict[Theme, dict[str, str]] = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e"},
    "dark": {"surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7"},
}
_MUTED = "#898781"  # the same muted ink on both themes
_LIGHT_INK, _DARK_INK = "#ffffff", "#0b0b0b"

_LABEL_W, _CELL_W, _CELL_H, _GAP = 250, 112, 30, 2
_TOP, _HEADER_H, _FOOT_H, _PAD = 74, 34, 44, 20


@dataclass(frozen=True)
class Cell:
    hits: int
    total: int

    @property
    def rate(self) -> float:
        return self.hits / self.total


@dataclass(frozen=True)
class ChartRow:
    label: str
    cells: dict[str, Cell]  # keyed by "all" and each recall category
    emphasis: bool = False


def recall_cells(outcomes: list[QuestionOutcome]) -> dict[str, Cell]:
    """Hits over questions with gold sources, overall and per category."""
    cells: dict[str, Cell] = {}
    for key in ("all", *RECALL_CATEGORIES):
        hits = [
            o.gold_hit
            for o in outcomes
            if o.gold_hit is not None and (key == "all" or o.category == key)
        ]
        if hits:
            cells[key] = Cell(sum(hits), len(hits))
    return cells


def fill_for(rate: float, theme: Theme) -> str:
    ramp = _RAMP if theme == "light" else tuple(reversed(_RAMP))
    return ramp[min(int(rate * len(ramp)), len(ramp) - 1)]


def _luminance(hex_color: str) -> float:
    def channel(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def ink_for(fill: str) -> str:
    """The ink that reads better on `fill`."""
    return max((_LIGHT_INK, _DARK_INK), key=lambda ink: contrast(fill, ink))


def build_svg(rows: list[ChartRow], theme: Theme, *, footnote: str) -> str:
    t = _THEMES[theme]
    columns = ("all", *RECALL_CATEGORIES)
    width = _PAD * 2 + _LABEL_W + len(columns) * (_CELL_W + _GAP)
    grid_top = _TOP + _HEADER_H
    height = grid_top + len(rows) * (_CELL_H + _GAP) + _FOOT_H
    brighter = "darker" if theme == "light" else "brighter"
    font = "font-family=\"-apple-system,'Segoe UI',Helvetica,Arial,sans-serif\""

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Recall@5 by question category, per configuration</title>',
        f'<desc id="desc">Share of answerable questions with a gold source in the top five '
        f"retrieved chunks, for {len(rows)} configurations of the ablation matrix.</desc>",
        f'<rect width="{width}" height="{height}" fill="{t["surface"]}"/>',
        f"<g {font}>",
        f'<text x="{_PAD}" y="30" font-size="17" font-weight="600" fill="{t["primary"]}">'
        "Recall@5 by question category</text>",
        f'<text x="{_PAD}" y="52" font-size="12.5" fill="{t["secondary"]}">Questions with a '
        f"gold source in the top 5 retrieved chunks. 10 questions per category; {brighter} "
        "is higher.</text>",
    ]
    for i, key in enumerate(columns):
        x = _PAD + _LABEL_W + i * (_CELL_W + _GAP) + _CELL_W / 2
        out.append(
            f'<text x="{x:.0f}" y="{grid_top - 12}" font-size="12" text-anchor="middle" '
            f'fill="{t["secondary"]}">{escape(COLUMN_TITLES[key])}</text>'
        )
    for r, row in enumerate(rows):
        y = grid_top + r * (_CELL_H + _GAP)
        weight = ' font-weight="600"' if row.emphasis else ""
        out.append(
            f'<text x="{_PAD + _LABEL_W - 12}" y="{y + _CELL_H / 2 + 4.5:.1f}" font-size="13" '
            f'text-anchor="end" fill="{t["primary"]}"{weight}>{escape(row.label)}</text>'
        )
        for i, key in enumerate(columns):
            x = _PAD + _LABEL_W + i * (_CELL_W + _GAP)
            cell = row.cells.get(key)
            if cell is None:
                out.append(
                    f'<text x="{x + _CELL_W / 2:.0f}" y="{y + _CELL_H / 2 + 4.5:.1f}" '
                    f'font-size="12" text-anchor="middle" fill="{_MUTED}">n/a</text>'
                )
                continue
            fill = fill_for(cell.rate, theme)
            label = f"{cell.hits}/{cell.total}"
            out.append(
                f"<g><title>{escape(row.label)}, {escape(COLUMN_TITLES[key])}: {label} "
                f"({cell.rate:.3f})</title>"
                f'<rect x="{x}" y="{y}" width="{_CELL_W}" height="{_CELL_H}" rx="4" fill="{fill}"/>'
                f'<text x="{x + _CELL_W / 2:.0f}" y="{y + _CELL_H / 2 + 4.5:.1f}" font-size="13" '
                f'text-anchor="middle" fill="{ink_for(fill)}"{weight}>{label}</text></g>'
            )
    out.append(
        f'<text x="{_PAD}" y="{height - 18}" font-size="11.5" fill="{_MUTED}">'
        f"{escape(footnote)}</text>"
    )
    out += ["</g>", "</svg>", ""]
    return "\n".join(out)
