"""Reciprocal Rank Fusion -- merge several independent rankings of the
same items into one, without their scores having to be comparable.

Pure: it takes lists of ids and returns `(id, fused_score)` pairs. The
retrieval service builds the id lists from the vector / full-text /
trigram branches, fuses them here, then re-hydrates the winning ids back
to `RetrievedChunk`.

    score(d) = Sum_i  w_i / (k + rank_i(d))

- `rank_i(d)` is d's 1-based position in ranking i; a ranking that does
  not contain d contributes nothing for it.
- `k` dampens how much the top positions dominate. A larger k flattens
  the curve, so agreement across rankings outweighs any single ranking's
  #1. `k = 60` is the value from the original RRF paper (Cormack, Clarke
  & Buettcher, 2009) and the Day 8 default.
- `w_i` weights ranking i; it defaults to 1.0 for every ranking. `k` and
  the weights are experiment-matrix variables in Week 4, which is why
  they are plain arguments here rather than hard-coded.
"""


def rrf(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse `rankings` (each a list of ids, best first) into one ranking.

    Returns `(id, score)` pairs sorted by score descending. Ties are
    broken by the id's first appearance across the input rankings, so the
    output order is deterministic for a given input.

    - `k` must be positive.
    - `weights`, when given, must have exactly one entry per ranking.
    - A repeated id within one ranking counts only at its first (best)
      position in that ranking.
    - An empty ranking is allowed and contributes nothing -- the trigram
      branch is empty whenever the question names no code.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if weights is None:
        weights = [1.0] * len(rankings)
    elif len(weights) != len(rankings):
        raise ValueError(
            f"weights has {len(weights)} entries, expected one per ranking "
            f"({len(rankings)})"
        )

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for weight, ranking in zip(weights, rankings, strict=True):
        counted: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in counted:
                continue
            counted.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            first_seen.setdefault(doc_id, len(first_seen))

    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
