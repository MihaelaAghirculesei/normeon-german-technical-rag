"""Detect when the retrieved sources disagree by version on the same
requirement code (plan, Giorno 13).

`Source.version_label` is a per-*document* value (one `Document` row has
exactly one `version_label`), so grouping conflicts by `document_id`
could never fire -- every chunk under one document_id already shares
that one label. What the spec's "two version_labels of the same
document" scenario actually looks like in this schema is two *different*
documents that are versions of one another (e.g. the corpus's synthetic
`Lastenheft-EPS-v1.2.docx` / `-v2.0.docx`, which deliberately restate
requirement code `LH-3.2.1` with different values) -- nothing in the
schema ties two `Document` rows together as "the same logical document"
except the requirement code their chunks both mention. So conflicts are
grouped by requirement code, not by document.

Pure: no I/O, no settings, no logging. Same split as
`domain/context.build_context` and `domain/citations.extract_and_validate`
-- the caller (`services/generation.py`) decides what a conflict means
for the prompt.
"""

from __future__ import annotations

from app.domain.context import Source
from app.domain.normalization import extract_all_codes


def find_version_conflicts(sources: dict[str, Source]) -> dict[str, list[str]]:
    """Scan every source's content for requirement codes and return only
    the codes referenced by sources carrying at least two distinct
    (non-`None`) `version_label`s, each mapped to its distinct labels in
    first-appearance order."""
    labels_by_code: dict[str, list[str]] = {}
    for source in sources.values():
        if source.version_label is None:
            continue
        for code in extract_all_codes(source.content):
            labels = labels_by_code.setdefault(code, [])
            if source.version_label not in labels:
                labels.append(source.version_label)

    return {code: labels for code, labels in labels_by_code.items() if len(labels) >= 2}
