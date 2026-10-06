# ADR 0003 — Structural chunking as the default

## Status
Accepted. Retrieval numbers are post-fix; answer-accuracy numbers come
from the run before the context-dedup fix and are refreshed when the
post-fix answer run completes (see `docs/EVALUATION.md`).

## Context
Every document is ingested with two chunking strategies side by side:
- **`fixed_500`**: 500-token windows with 15% overlap over the whole
  document's text, blind to its structure;
- **`structural`**: one chunk per leaf section, small sections merged,
  long ones split, the section breadcrumb prepended to the content.

Retrieval, citations and the answer prompt all run over one of them; the
Giorno 20 matrix measured both, crossed with vector-only and hybrid
retrieval, on the 50-question eval set (10 per category).

| Config | recall@5 (post-fix) | answer_accuracy (run 1) | false abstention (run 1) |
| --- | --- | --- | --- |
| structural / hybrid | **0.925** [0.825, 1.000] | 0.780 [0.660, 0.880] | 0.250 |
| structural / vector | 0.875 [0.775, 0.975] | 0.760 [0.640, 0.880] | 0.300 |
| fixed_500 / hybrid | 0.575 [0.425, 0.725] | 0.860 [0.760, 0.940] | 0.150 |
| fixed_500 / vector | 0.575 [0.425, 0.725] | **0.880** [0.780, 0.960] | 0.175 |

95% bootstrap intervals in brackets. Paired against structural / hybrid,
fixed_500's recall deficit is significant (−0.225, [−0.375, −0.075]);
its accuracy lead is not (+0.08, [−0.04, +0.20]).

## Decision
Keep `structural` as the default `retrieval_strategy`, with hybrid
retrieval. Treat the generator's over-abstention as a separate problem to
fix in the prompt and context, not by changing the chunker.

## Reasoning
- **The one difference the data resolves favours structural.** Recall is
  35 points higher after the fix (22.5 before it, paired interval clear of
  zero); the accuracy lead of
  fixed_500 is 8-10 points with intervals that include zero. A default
  should follow the measured effect, not the larger point estimate.
- **Citations need the right section.** A fixed window spans several
  sections but is labelled with the first one it starts in -- for each
  Lastenheft, the document title. 15 of fixed_500's 17 recall misses are
  this label, not missing text, and the same label is what a citation
  would show the user. The product promise is "from the answer to the
  exact place in the source"; a structural chunk's label is correct by
  construction.
- **The accuracy gap has a visible mechanism that chunking won't fix
  well.** Nine of the baseline's wrong answers abstained with a gold
  source in context. Wider windows reduce that as a side effect of
  carrying more text; the targeted fix is to give the generator that
  neighbouring context (or relax the prompt) while keeping precise
  sources.
- **Version comparison is section-shaped.** The conflict questions ask
  how section X differs between two versions; structural chunks align
  with that unit (once context dedup is keyed per document -- the bug the
  run found).

## Consequences
- The generator's false-abstention rate (0.25 on the baseline) is the
  next quality lever: a prompt or context change, measured against this
  same matrix.
- If the post-fix answer run, or the Giorno 28 ablation, shows fixed_500
  ahead on accuracy with an interval clear of zero *after* the abstention
  fix, revisit this decision. The two strategies stay ingested side by
  side, so switching is a configuration change.
- Fixed windows should not be used for citations until they record every
  section they span; until then their recall is reported as a lower
  bound.
- Fifty questions bound how small a difference can be detected (about
  ±10 points of accuracy). Decisions that hinge on smaller effects need a
  larger eval set first.
