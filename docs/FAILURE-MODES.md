# Failure modes

An honest catalogue of where this system is known to be weak, why, and
what (if anything) compensates. Grows as each stage is built and measured;
the evaluation in Week 4 will attach numbers to several of these.

## Full-text search

### German compound nouns are not decomposed

PostgreSQL's `german` snowball stemmer stems but does not split compounds.
`Fahrzeugzulassung` becomes the lexeme `fahrzeugzulass`, which does **not**
match a query for `Zulassung` (`zulass`):

```sql
SELECT to_tsvector('german', 'Fahrzeugzulassung')
       @@ websearch_to_tsquery('german', 'Zulassung');   -- => false
```

German legal and technical text is full of these (`Betriebserlaubnis`,
`Hauptuntersuchung`, `Rückhalteeinrichtung`, `Beleuchtungseinrichtung`),
so a full-text query phrased with a different compounding than the source
text silently returns nothing.

**Not fixed.** Third-party decompounders (e.g. a CompoundWordTokenFilter
port, or a dictionary splitter) were considered and rejected for this
project: they need a maintained German dictionary, misfire on domain
terms, and add a dependency whose failures are hard to see. Recognising
and measuring the limit is worth more here than papering over it.

**Mitigation.** The vector branch is not lexical and handles compound
paraphrase well (verified in `scripts/try_fts_search.py`: "Wie hell
müssen die Frontleuchten strahlen?" returns nothing from FTS but the
right `§ 50` chunk from vector). Day 8's RRF fusion runs both branches so
the vector side covers the FTS side's blind spot and vice versa.

### tsquery AND semantics punish paraphrased questions

`websearch_to_tsquery` ANDs the query terms. A natural-language question
("Wann muss ein Auto zum TÜV?") contributes terms the corpus never uses
verbatim ("Auto", "TÜV" vs. the statute's "Kraftfahrzeug",
"Hauptuntersuchung"), and one missing term drops the whole match. FTS
here is a keyword / code-reference tool, not a question-answering tool.

**Mitigation.** Same as above — the vector branch, and hybrid fusion on
Day 8. `normalize_de` canonicalises requirement codes and norm references
(`LH-3.2.1`, `UN R79`) into single tokens precisely so the one thing FTS
*should* be reliable at — exact reference lookup — is not defeated by
tokenisation.

## Hybrid retrieval (RRF fusion)

`hybrid_search` fuses the vector, full-text and trigram rankings with
Reciprocal Rank Fusion: `score(d) = Σ_i w_i / (k + rank_i(d))`.

### RRF sees rank, not confidence

Fusion uses each hit's *position* in its branch, not its branch score. A
vector hit at cosine `0.92` and one at `0.55` contribute the same amount
if both are rank 1. So a branch that is confidently right and a branch
that is weakly guessing count equally per position; the only lever is the
per-branch weight, which is global, not per-query.

### Consensus can outweigh a single strong signal

A chunk found by one branch only is capped at that branch's contribution
(`w_i / (k + rank)` ≈ `0.016` at rank 1 with the default `k = 60`), while
a chunk sitting at rank 3 in two branches scores ≈ `0.031`. This is
usually the behaviour we want, but it can push down a chunk that only the
vector branch recognised as on-topic when the lexical branches had
nothing relevant to say.

**Mitigation / status.** `k` and the three branch weights
(`rrf_k`, `rrf_weight_vector` / `_fts` / `_trgm` in `core/config.py`) are
deliberately parameters, not constants. Their defaults (`k = 60`, equal
weights) are placeholders; the Week 4 experiment matrix tunes them
against the 50-question eval set, and the numbers land here.

## Reranking + context selection

`retrieve_context` runs the hybrid candidates through a cross-encoder
(`bge-reranker-v2-m3`) and then fills a token budget, skipping a
`section_path` already represented.

### The cross-encoder truncates long chunks

The reranker model has a fixed maximum input length (~512 tokens for the
`(query, chunk)` pair). A structural chunk near the 800-token ceiling is
truncated before scoring, so its tail never influences its rerank score —
the relevant sentence may be the part that got cut.

### Section dedup is first-chunk-wins

Context selection keeps only the first chunk of any `section_path`. If the
answer is split across two adjacent chunks of one long section, the
second is dropped even when it would have fit the budget.

### The token budget is approximate

The 4000-"token" budget counts whitespace words, not the generator's
subword tokens (German runs ~1.3× more subword tokens than words). It is
a relevance/diversity cap, not a hard context-window guarantee — the
generation day will need its own real-tokenizer check.

**Cost — the cross-encoder is not viable live on the dev box.**
`scripts/try_rerank.py` measured `bge-reranker-v2-m3` at **~10 s per
`(query, chunk)` pair** on this CPU (xlm-roberta-large, fp32, a RAM-tight
machine also running the DB). Over the 40 hybrid candidates that is ~7
minutes per request — the plan's "~1.5 s for 40 pairs" assumes a smaller
or GPU-served reranker. So on this hardware the practical config is
`reranker_provider = "noop"` (or a much smaller `hybrid_candidate_k`);
`cross_encoder` stays the real implementation for the Week 4 matrix,
which will need a GPU or a served reranker. The pipeline itself is
correct — `retrieve_context` and the adapters are covered by tests with
a fast fake reranker, and the cross-encoder's *ranking* on real chunks
is sound (it lifted the substantive § 41 braking rule over § 42 for
"Bremse Anhänger", § 36 tyres from #3 to a near-tie for #1 for "Reifen
Profiltiefe").

## Generation / citation validation

`domain/citations.extract_and_validate` checks that every `[S..]` marker
the model wrote is one it was actually offered. That is a *presence*
check, not a *correctness* check.

### A citation can be present and still wrong

Citing a real, offered `[S1]` proves the marker exists — not that the
sentence in front of it is actually supported by `S1`'s content. A model
can attach a correct-looking marker to a claim that chunk does not
support (misattribution) or that is a subtly wrong paraphrase of it. The
validator cannot see that; only a stricter grounding check (e.g. an
LLM-judge pass in the Week 4 evaluation) can.

**Mitigation / status.** Out of scope for a marker-presence check by
construction. The Week 4 evaluation suite's faithfulness judge is where
this gets measured; `extract_and_validate` is the cheap, reliable first
filter (catches every fabricated *reference*, i.e. a marker pointing at
nothing) that a semantic check would sit behind, not replace.

### The fake LLM needed a real fix, not a per-test workaround

`FakeLlmClient` originally scanned the *whole* rendered prompt for
`[S\d+]` markers. The prompt's own rule text unconditionally contains
the literal `[S1], [S2], ...`, so whenever real retrieval returned zero
or exactly one chunk, the fake "cited" a marker that was never actually
offered — indistinguishable from a real hallucinated citation once
`extract_and_validate` existed to catch it. Fixed by scanning only the
context block between the prompt's `Quellen:` and `Frage:` markers
(`adapters/llm/fake.py`); this is inherent coupling of a test double to
the one prompt family it approximates, not a citation-validation gap.

## Parsing / chunking

Carried over from earlier days, revisit if Week 4 eval shows they matter:

- The numbered-heading heuristic false-positives on numeric data inside
  annex tables that `find_tables()` misses (Day 3 notes).
- Cascading merges of tiny table-of-contents lines produce one large
  low-value chunk; a section just over the split threshold can leave a
  sub-100-token runt piece (Day 4 notes).
- No real character offsets into the source PDF yet (`char_start`/
  `char_end` are `0`/`len`), so highlight-in-original is not possible
  until that is tracked.
