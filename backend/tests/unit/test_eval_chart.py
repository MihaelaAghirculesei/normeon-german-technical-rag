import xml.etree.ElementTree as ET

import pytest

from app.eval.chart import (
    _RAMP,
    _THEMES,
    Cell,
    ChartRow,
    build_svg,
    contrast,
    fill_for,
    ink_for,
    recall_cells,
)
from app.eval.metrics import QuestionOutcome

SVG = "{http://www.w3.org/2000/svg}"


def _outcome(category: str, gold_hit: bool | None) -> QuestionOutcome:
    return QuestionOutcome.model_validate(
        {"question_id": "Q001", "category": category, "answer_correct": True, "gold_hit": gold_hit}
    )


def test_recall_cells_count_only_questions_with_gold_sources() -> None:
    outcomes = [
        _outcome("conflict", True),
        _outcome("conflict", False),
        _outcome("code_lookup", True),
        _outcome("unanswerable", None),
    ]

    cells = recall_cells(outcomes)

    assert cells["all"] == Cell(2, 3)
    assert cells["conflict"] == Cell(1, 2)
    assert cells["code_lookup"] == Cell(1, 1)
    assert "unanswerable" not in cells
    assert "requirement_lookup" not in cells  # no question, no cell


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a_higher_rate_never_moves_away_from_the_strong_end(theme: str) -> None:
    surface = _THEMES[theme]["surface"]  # type: ignore[index]
    fills = [fill_for(rate / 20, theme) for rate in range(21)]  # type: ignore[arg-type]
    distances = [contrast(fill, surface) for fill in fills]

    assert distances == sorted(distances)  # more recall, further from the surface
    assert fills[0] != fills[-1]


@pytest.mark.parametrize("fill", _RAMP)
def test_cell_text_stays_readable_on_every_fill(fill: str) -> None:
    assert contrast(fill, ink_for(fill)) >= 4.5


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_svg_is_valid_and_prints_every_value(theme: str) -> None:
    rows = [
        ChartRow("Baseline & co <x>", {"all": Cell(37, 40), "conflict": Cell(8, 10)}, True),
        ChartRow("E3", {"all": Cell(6, 40)}),
    ]

    svg = build_svg(rows, theme, footnote="note")  # type: ignore[arg-type]
    texts = [el.text for el in ET.fromstring(svg).iter(f"{SVG}text")]

    assert "Baseline & co <x>" in texts  # escaped in the file, intact once parsed
    assert {"37/40", "8/10", "6/40"} <= set(texts)
    assert texts.count("n/a") == 3 + 4  # missing categories are marked, not left blank
