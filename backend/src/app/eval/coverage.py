"""Where an abstained question's answer actually was.

A false abstention -- `NICHT_GEFUNDEN` on an answerable question -- has
two possible causes: the context the model saw lacked the answer, or it
had the answer and declined anyway. Adding each chunk's neighbours
(`ordinal` +-1) can only fix the first kind, and only when the missing
answer point sits in a neighbour. `locate_point` sorts each expected
answer point into one of three places, with Layer 1's matching rule, so
the most that neighbour expansion could fix is known before paying for
an answer run.

Pure -- the caller supplies the texts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from app.eval.scoring import normalize_for_match

PointLocation = Literal["context", "neighbour", "missing"]


def locate_point(point: str, context: str, neighbours: str) -> PointLocation:
    """Where `point` is: in the `context` the model saw, only in an
    adjacent chunk (`neighbours`), or "missing" from both."""
    needle = normalize_for_match(point)
    if needle in normalize_for_match(context):
        return "context"
    if needle in normalize_for_match(neighbours):
        return "neighbour"
    return "missing"


def neighbours_could_answer(locations: Sequence[PointLocation]) -> bool:
    """True when neighbour expansion would complete the context: every
    point is in the context or a neighbour, and at least one only in a
    neighbour. A question whose points are all in the context already is
    a decline the context cannot fix."""
    return "missing" not in locations and "neighbour" in locations
