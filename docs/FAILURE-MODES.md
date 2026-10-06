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

Each `(query, chunk)` pair is capped at `reranker_max_length` tokens (512,
the model authors' own default). A structural chunk near the 800-word
ceiling — roughly 1,000+ subword tokens of German — is truncated before
scoring, so its tail never influences its rerank score; the relevant
sentence may be the part that got cut.

Until Giorno 20 this paragraph described a cap the code did not apply:
no `max_length` was passed, so sentence-transformers fell back to the
tokenizer's own limit (8192 for `bge-reranker-v2-m3`) and every chunk was
scored in full. That was more faithful but considerably slower — part of
the per-pair cost measured below. The cap is now explicit, configurable
and recorded in each eval run's config (`EvalConfig.reranker_max_length`),
so its quality cost is something the matrix can measure instead of assume.

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

## Abstention and conflict detection (Day 13)

### The confidence threshold's scale depends on the reranker

`min_rerank_score_for_answer` compares against `retrieve_context`'s top
reranked chunk score, but that score's scale is whatever the configured
`reranker_provider` produces: `cross_encoder`'s raw logits are an
unbounded real number (roughly -10..10 for `bge-reranker-v2-m3`,
uncalibrated), while `noop` just passes through the hybrid stage's
RRF-fused score, which is *always* a small positive number regardless of
true relevance (`w / (k + rank)`, see the RRF entry above). A threshold
tuned for one is meaningless for the other -- with `noop` a real
NICHT_GEFUNDEN case still clears any reasonable threshold above zero,
because RRF never really says "not confident", only "not top-ranked".

**Mitigation / status.** Same "Week 4 matrix variable" status as `rrf_k`
and the rerank weights: `min_rerank_score_for_answer` needs per-reranker
tuning against the eval set, not a single one-size value. The gate is
correct in what it does (skip the LLM call below the configured number);
what number is meaningful is a `reranker_provider`-specific question.

### Version-conflict detection is keyed on the requirement code, not the document

The spec's scenario ("chunks from two `version_label`s of the same
document") does not map onto this schema directly: `version_label` is a
per-`Document` value, so every chunk under one `document_id` already
shares it -- grouping by `document_id` could never find a conflict. The
corpus's own synthetic conflict case (`Lastenheft-EPS-v1.2.docx` /
`-v2.0.docx`, both restating requirement code `LH-3.2.1` with different
values, see `corpus/manifest.yaml`) is two *different* `Document` rows
that are versions of one another -- nothing in the schema ties them
together except that shared code. So `domain/conflicts.
find_version_conflicts` groups by requirement code
(`domain.normalization.extract_all_codes`) extracted from chunk content,
not by document identity.

**Consequence.** A real disagreement between two document versions that
does *not* restate a matching requirement-code-shaped token in both
chunks' text is invisible to this check -- e.g. prose that changes a
requirement without repeating its code, or two versions that use
different code spellings normalisation doesn't unify. Fixing this
properly needs an explicit "these Document rows are versions of one
another" link in the schema (a `document_group`/`supersedes` field),
which is out of scope for Day 13; revisit if the Week 4 conflict eval
case needs more recall than the code-matching heuristic gives it.

## Streaming (Day 14)

`POST /api/v1/chat/stream` sends `token` events as the model produces
them, and only runs citation validation once the whole answer has been
assembled.

### A token already streamed cannot be un-sent

If the model writes an invented marker (`[S9]`), the client has already
seen it in a `token` event by the time `extract_and_validate` runs on
the complete text. The `done` event's `citations` is still the honest,
validated list -- it simply may not match everything the client already
rendered from the raw token stream. The non-streaming `POST /api/v1/chat`
does not have this problem: it validates before returning anything.

**Mitigation / status.** Not fixed by construction -- buffering the
whole answer before streaming any of it would remove the point of
streaming (perceived responsiveness, `sources` arriving before the
text). A frontend that wants to be defensive here can treat `done.
citations` as authoritative and visually flag/retract a rendered answer
whose citations came back empty; that UI behaviour is a later
(Angular, Week 5) concern, not something the backend can paper over.

### The LLM adapter's stream() does not retry

`OpenAICompatibleClient.complete` retries transient failures (Day 13);
`stream` deliberately does not, because retrying after even one chunk
has already reached the caller would either duplicate output or need
"has anything been yielded yet" bookkeeping that was not judged worth
the complexity for Day 14. A transient mid-stream failure becomes one
`error` SSE event instead of a retried attempt.

## Cost tracking (Day 15)

### pricing.yaml is a snapshot, not a live feed

`services.pricing.calculate_cost` looks up a fixed table
(`backend/pricing.yaml`) verified against a pricing aggregator on the
date noted in that file's header. Provider prices change without
notice and this file is never re-fetched automatically -- a cost
reported today against a stale entry is only as accurate as the last
time someone re-verified it. An unpriced model (anything not listed,
including the `fake` provider and any self-hosted model with a
homegrown id) returns `cost_usd = null` rather than a guess, which is
correct but means "no cost recorded" cannot be told apart from "this
answer was free" from the number alone -- `query_logs.tokens` still
records the token counts either way.

### Streaming cost depends on the server supporting `stream_options.include_usage`

`OpenAICompatibleClient.stream` asks for a final usage-only delta via
`stream_options: {"include_usage": true}` -- an OpenAI extension most,
but not all, OpenAI-compatible servers honour (some vLLM/TGI/llama.cpp
deployments silently ignore unknown request fields rather than
erroring, so this degrades quietly). If the server doesn't send it,
`Delta.prompt_tokens`/`completion_tokens` stay `None` for the whole
stream and the streamed answer's `cost_usd`/`query_logs.tokens` come
back empty, exactly like a `complete()` response from a server that
doesn't report `usage` at all -- not a bug, just no signal to work
with.

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

## Evaluation findings (Giorno 20)

The first full matrix run -- `{structural, fixed_500} x {hybrid, vector}`,
cross-encoder reranker, 10 candidates, `top_k=5`, 50 questions -- is in
`docs/EVALUATION.md`. Five things it showed, each traced to the
questions that produced it rather than read off an aggregate.

### 1. Context dedup dropped the second version of a section (fixed)

`_select_context` skipped a chunk whose `section_path` was already in the
context *regardless of document*. Lastenheft v1.2 and v2.0 share section
numbers, so whenever v1.2's section 5 ranked first, v2.0's section 5 --
ranked second by the reranker -- was discarded as a duplicate. Five
questions about v2.0 (Q011-Q014, Q037) were answered from the wrong
version. The dedup key is now `(document_id, section_path)`; re-measured
retrieval-only, baseline recall@5 went from **0.800 to 0.925**. This is
the version-conflict case the corpus was built to test, and it was
failing in context selection, not in retrieval.

### 2. The generator over-abstains on narrow structural context

Ten of the baseline's eleven wrong answers are `NICHT_GEFUNDEN`, and in
nine of them a gold source *was* in the context. These are not the
pre-generation confidence gate (the LLM was called each time). With wider
fixed windows seven of those ten are answered: false-abstention rate 0.25
(structural / hybrid) vs 0.15 (fixed_500 / hybrid). The likely cause --
a hypothesis, not yet isolated -- is that the answer needs a sentence of
surrounding text a single structural chunk doesn't carry, and the strict
"answer only from the sources" prompt makes the model decline. Accuracy is limited by generation,
not retrieval -- the lever is the prompt and the context window (e.g.
adding the neighbouring chunk), not the chunker.

### 3. Fixed windows carry the wrong section label

A fixed 500-token window spans several sections but is stored with the
`section_path` of the first heading it starts in; for each Lastenheft it
is the document title. 15 of fixed_500's 17 recall misses are Lastenheft
questions whose gold source names a section only -- the text is in the
chunk, the label is not. That makes fixed_500's recall a lower bound, but
it is also a real product defect: a citation built from that chunk shows
the user the wrong section. On PDF-sourced questions the two strategies
are close (fixed_500 misses 2, structural 0).

### 4. A change log outranks the section it describes

For Q003 and Q004 the reranker puts "Anhang A: Änderungshistorie" -- the
change log that *mentions* the changed values -- above sections 6 and
3.2, which actually state them, and the real sections fall outside the
top five. The answer can still be right (the log contains the numbers),
but the citation points at the log, not the requirement. A cross-encoder
rewards lexical overlap with the question; a summary of changes overlaps
more than the requirement itself.

### 5. Fifty questions cannot separate configurations on accuracy

Every paired difference in `answer_accuracy` has a 95% interval that
includes zero (e.g. fixed_500 / hybrid vs. baseline: +0.08, [-0.04,
+0.20]). Recall differences can be resolved (structural vs. fixed_500:
-0.225, [-0.375, -0.075]); accuracy ones cannot at this sample size.
Conclusions about answer quality from this run are directional only.

### Operational: the run itself

- **No overall deadline on an LLM call.** httpx's `timeout` bounds each
  read, not the request: one answer took 277 s, and when the laptop
  entered modern standby a connection hung for nine hours with the
  process idle. The adapter needs a total per-call deadline.
- **Free-tier quota is per day, not just per minute.** Retries and pacing
  handle the per-minute limit; Gemini's free tier also caps a model at
  500 requests a day, and one full matrix (200 answers + 200 judgements)
  uses most of it. The post-fix accuracy re-run waits for the reset
  rather than switching models mid-experiment, which would make the
  before/after numbers incomparable.
