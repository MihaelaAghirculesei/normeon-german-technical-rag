# ADR 0002 — Reciprocal Rank Fusion instead of score normalisation

## Status
Accepted

## Context
Hybrid retrieval runs three branches against the same tenant's chunks and
has to merge their results into one ranking:
- **vector**: cosine similarity, bounded `[-1, 1]`, in practice `~0.75-0.90`
  for anything on-topic — a narrow, high band;
- **full-text**: `ts_rank_cd`, unbounded above, typically `0.0-0.3`, and
  very sensitive to term frequency and proximity;
- **trigram**: `word_similarity`, bounded `[0, 1]`, only present when the
  question names a code.

These scores are not on the same scale and are not calibrated against each
other. Some way to combine them is needed.

## Decision
Fuse the branches with Reciprocal Rank Fusion (Cormack, Clarke & Buettcher,
2009): `score(d) = Σ_i w_i / (k + rank_i(d))`, using each hit's **rank**
within its branch, not its branch score. `k` (default 60) and the
per-branch weights (default 1.0) are configuration, tuned by the Week 4
experiment matrix.

## Reasoning
- **The branch scores are not comparable.** Min-max or z-score
  normalisation makes them numerically comparable but not *meaningfully*
  so — a `ts_rank_cd` of 0.15 and a cosine of 0.85 both normalising to
  ~0.9 does not mean the two hits are equally good.
- **Normalisation is unstable across queries.** Its output depends on the
  min and max present in each branch's candidate set. One unusually strong
  hit compresses the rest toward zero; a branch that returns nothing
  relevant still gets its top hit normalised to 1.0. RRF has neither
  problem: an absent branch simply contributes nothing, and one strong hit
  does not distort the others.
- **Rank order is the one thing every branch produces reliably.** Even
  when a branch's absolute scores are noise, its ordering carries signal.
- **One interpretable knob.** `k` controls how sharply top ranks dominate;
  the weights control per-branch influence. Both map directly onto
  experiment-matrix axes. A normalisation scheme would add per-branch
  choices (which normaliser, clipping, outlier handling) that are harder
  to reason about and to sweep.
- **It is the retrieval-literature default** for exactly this
  score-incompatibility problem, which makes results comparable to
  published work.

## Consequences
- RRF discards confidence: a branch that is *certain* and a branch that is
  *guessing* count the same at the same rank. This is a real limitation,
  documented in `docs/FAILURE-MODES.md` ("Hybrid retrieval (RRF fusion)").
- If the Week 4 matrix shows a normalised or learned fusion beating RRF on
  the 50-question eval set, this decision should be revisited — the
  branches already return `RetrievedChunk` objects carrying their branch
  score, so a score-aware fusion is a drop-in replacement for `rrf()` in
  `app.services.retrieval.hybrid`.
