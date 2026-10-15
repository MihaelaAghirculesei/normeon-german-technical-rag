"""Answer prompts are versioned files under ``backend/prompts/``, loaded
by name. Kept as files (not inline strings) so a prompt change is a
reviewable diff with its own history, and so the exact text -- by name
*and* content hash -- can be logged with every answer it produced
(plan, Giorno 11).

The template carries the sources and ``{{QUESTION}}``, filled by
:meth:`Prompt.render`. The sources go in as ``{{CONTEXT}}`` (v1: a plain
block) or ``{{QUELLEN}}`` (v2 onwards: each source delimited and marked
as untrusted data, see ``domain/context.build_context``). No
``str.format`` / ``%``: chunk text routinely contains braces and percent
signs, which would blow up or silently mangle it.

Substitution is one pass over the template, never chained replaces: a
document that contains the text ``{{QUESTION}}`` must stay that text,
not be rewritten into the user's question (Giorno 26).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parents[3] / "prompts"

CONTEXT_PLACEHOLDER = "{{CONTEXT}}"
QUELLEN_PLACEHOLDER = "{{QUELLEN}}"
QUESTION_PLACEHOLDER = "{{QUESTION}}"
_PLACEHOLDER = re.compile(r"\{\{(CONTEXT|QUELLEN|QUESTION)\}\}")


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    template: str
    sha256: str

    @property
    def untrusted_context(self) -> bool:
        """Whether this prompt expects the delimited, untrusted-data form
        of the sources (``{{QUELLEN}}``) rather than the v1 plain block."""
        return QUELLEN_PLACEHOLDER in self.template

    def render(self, *, context: str, question: str) -> str:
        values = {"CONTEXT": context, "QUELLEN": context, "QUESTION": question}
        return _PLACEHOLDER.sub(lambda m: values[m.group(1)], self.template)


@lru_cache
def load_prompt(name: str) -> Prompt:
    """Read ``backend/prompts/<name>.txt`` and hash it. Cached: the set of
    prompts is small and fixed for the life of the process."""
    text = (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Prompt(name=name, template=text, sha256=digest)
