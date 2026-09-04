"""Answer prompts are versioned files under ``backend/prompts/``, loaded
by name. Kept as files (not inline strings) so a prompt change is a
reviewable diff with its own history, and so the exact text -- by name
*and* content hash -- can be logged with every answer it produced
(plan, Giorno 11).

The template carries two placeholders, ``{{CONTEXT}}`` and
``{{QUESTION}}``, filled by :meth:`Prompt.render`. Plain ``str.replace``
is used on purpose: chunk text routinely contains braces and percent
signs, so ``str.format`` / ``%`` would blow up or silently mangle it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parents[3] / "prompts"

CONTEXT_PLACEHOLDER = "{{CONTEXT}}"
QUESTION_PLACEHOLDER = "{{QUESTION}}"


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    template: str
    sha256: str

    def render(self, *, context: str, question: str) -> str:
        return self.template.replace(CONTEXT_PLACEHOLDER, context).replace(
            QUESTION_PLACEHOLDER, question
        )


@lru_cache
def load_prompt(name: str) -> Prompt:
    """Read ``backend/prompts/<name>.txt`` and hash it. Cached: the set of
    prompts is small and fixed for the life of the process."""
    text = (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Prompt(name=name, template=text, sha256=digest)
