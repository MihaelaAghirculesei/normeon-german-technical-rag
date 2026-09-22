# Evaluation

Status: work in progress (plan, Giorni 17-20). This document tracks the
50-question evaluation set, the two-layer correctness metric, and the
judge-agreement check. Sections below are filled in as each part lands.

**Progress: 50/50 questions written, all validated in CI** (`tests/unit/test_eval_schema.py::test_full_question_set_has_fifty_entries_ten_per_category`).
`requirement_lookup`, `cross_reference`, `code_lookup`, `unanswerable` and
`conflict` are each 10/10.

Blocker found and fixed before writing `conflict`: the two Lastenheft
`.docx` were never ingested (no DOCX parser existed), so the conflict
case would have been untestable. Added `adapters/parsing/docx.py` +
wired `version_label`/`valid_from`/`valid_until` through ingestion (see
git history on `day17-18/eval-question-set-and-judge`). Verified against
the real dev DB: a live retrieval query surfaces chunks from both
Lastenheft versions and `find_version_conflicts` correctly flags both
`LH-3.2.1` and `LH-6.2.1`.

Known limitation carried into the gold sources: the DOCX parser has no
page concept (short document, every block is stamped page 1), so
Lastenheft-sourced `gold_sources` cite `section` only, never `page`.

Real body-heading pages found in `corpus/StVZO.pdf` (via the parser, not
guessed — the first hits on these codes are the table of contents on
pages 2-4, not the article itself):

| Section | Title | Page |
|---|---|---|
| §32 | Abmessungen von Fahrzeugen und Fahrzeugkombinationen | 21 |
| §35a | Sitze, Sicherheitsgurte, Rückhaltesysteme, ... | 32 |
| §36 | Bereifung und Laufflächen | 35 |
| §41 | Bremsen und Unterlegkeile | 40 |
| §42 | Anhängelast hinter Kraftfahrzeugen und Leergewicht | 45 |
| §50 | Scheinwerfer für Fern- und Abblendlicht | 55 |
| §53d | Nebelschlussleuchten | 65 |
| §55 | Einrichtungen für Schallzeichen | 67 |

And in `corpus/FZV.pdf`:

| Section | Title | Page |
|---|---|---|
| §9 | Zuteilung von Kennzeichen | 1 |
| §10 | Besondere Kennzeichen | 2 |

## The question set (Giorni 17-18)

`backend/eval/dataset/questions.yaml` — 50 questions, hand-written by
reading `corpus/StVZO.pdf`, `corpus/FZV.pdf`,
`corpus/Lastenheft-EPS-v1.2.docx` and `corpus/Lastenheft-EPS-v2.0.docx`.
**Not LLM-generated**: a model asked to write questions from the same
chunks the retriever indexed would produce questions the system solves
by construction, making the evaluation meaningless.

Five categories, 10 questions each:

| Category | Rule |
|---|---|
| `requirement_lookup` | A single, specific fact. At least 5 of the 10 must cite a value that appears in a **table**, not prose. |
| `cross_reference` | Answering requires combining two sections, or two documents. |
| `code_lookup` | The question centers on a code, abbreviation, or norm number (e.g. `LH-3.2.1`, `UN R79 §5.1.2`) — the case where pure embedding search struggles and hybrid retrieval earns its place. |
| `unanswerable` | Not in the corpus. `should_abstain: true`. Must be **plausible** — close to a real topic but genuinely absent — not absurd; a question the system could mistake for answerable is the real test. |
| `conflict` | Lastenheft-EPS v1.2 and v2.0 contradict each other (both tied to requirement code `LH-3.2.1`). A correct answer surfaces the conflict rather than picking one version silently. |

Authoring checklist per question:

1. Open the actual source PDF/DOCX and locate the passage — do not rely on
   the chunk store or a paraphrase.
2. Write `id` (`Q001`..`Q050`, sequential), `category`, `question` (German).
3. `expected_answer_points`: the facts that MUST appear in a correct
   answer, phrased so a tolerant substring/number match can check them.
4. `gold_sources`: `document` + `section` and/or `page`, verified by
   reading that exact spot in the source file — not guessed from a
   section number seen elsewhere.
5. `should_abstain`: `true` only for `unanswerable`.
6. `difficulty`: `easy` / `medium` / `hard`.
7. `notes`: free text; use it to flag e.g. "value in table" for the
   `requirement_lookup` table-sourced questions, so the ratio is checkable
   later.

Validated against `EvalQuestion` (`src/app/eval/models.py`) via
`load_questions`; the full-set shape test (50 entries, 10 per category)
is `tests/unit/test_eval_schema.py::test_full_question_set_has_fifty_entries_ten_per_category`.

## Correctness metric (Giorno 18)

**Done.** Run for real against the full 50-question set
(`eval/reports/day18-merged-real-run.json` /
`day18-merged-real-run.scored.json`, gitignored — regenerate from the
DB + `.env` if lost, don't recommit). See "Judge model note" below for
the judge-quota issue hit and fixed along the way.

- Layer 1 — deterministic (`src/app/eval/scoring.py:score_layer1`):
  every `expected_answer_points` string must appear in the answer after
  normalizing only numeric *formatting* (decimal comma vs. dot,
  whitespace, case) — not a fuzzy/semantic match. Zero cost, zero LLM
  calls, fully reproducible.
- Layer 2 — LLM-as-judge (`score_layer2`): a versioned prompt
  (`prompts/eval_judge_de.v1.txt`) scores 0-2 given the question, the
  expected points, the gold sources, and the answer, returning strict
  JSON. A malformed judge response comes back as `JudgeError` data
  rather than raising.
- `score_report` runs both layers over every answer in an existing
  `EvalReport`. `scripts/score_eval_report.py` is the CLI: takes a
  report JSON, writes `<hash>.scored.json` and a
  `<hash>.human_review.csv` worksheet (see below).

## Judge agreement (Giorno 18)

**Done — all 15 pairs scored, percent agreement 1.0, Cohen's kappa
1.0.**

- `select_human_review_sample` (`src/app/eval/agreement.py`) picks 3
  questions per category (15 total, deterministic — the first 3 by id
  in each category), so every category is represented.
- `write_human_review_worksheet` exports those 15 as a CSV: question,
  the system's answer, the gold facts/sources, the judge's own score —
  and a blank `human_score` column. Written `utf-8-sig` + `;`-delimited
  so it opens correctly in Excel on a German/Italian-locale Windows
  install (a BOM-less, comma-delimited version silently lost 8 of 15
  rows when round-tripped through Excel on this machine — fixed).
- Human scoring was done by hand against the German question/answer/
  gold-source text directly (not translated — see the authoring
  checklist above), independently of the judge's own score column
  (blinded during scoring, compared only afterwards).
- **Result on all 15 pairs**: human and judge scores matched on all
  15 — **percent agreement 1.0, Cohen's kappa 1.0**
  (`eval/reports/day18-final-human-review.csv`, reproducible with
  `scripts/measure_judge_agreement.py`). Unlike the earlier 7/15
  partial measurement, this now includes the categories most likely to
  disagree (all 3 `code_lookup`, all 3 `unanswerable`, both
  false-abstention cases `Q021`/`Q023`), so this is the real number,
  not a preliminary one.

### Judge model note: quota exhaustion, worked around by switching models

8 of the 15 human-review-sample questions (`Q021`, `Q023`, `Q031`,
`Q032`, `Q033`, `Q041`, `Q042`, `Q043`) got `429 Too Many Requests`
from the judge call on `gemini-omni-1.1-flash`, **every single time,
across at least 5 separate attempts over 6 days** (2026-09-15, -16,
-21). Same 8 questions failed every time — not the random pattern a
per-minute rate limit would produce — confirming a **daily/persistent
quota exhausted for this specific model name** on this API key,
distinct from other Gemini models on the same key.

Fixed 2026-09-21 by switching the judge model to
`gemini-flash-lite-latest` (`backend/.env`, `LLM_MODEL`) — verified via
`GET /v1beta/openai/models` that this key has access to a wide set of
Gemini model names, and confirmed by direct call that `gemini-2.0-flash`
(the first guess) 404s for this key/account ("no longer available to
new users"), while `gemini-flash-lite-latest` answers normally with no
429. All 8 previously-blocked judge calls were re-run for real on this
model and succeeded (scores: `Q021`=0, `Q023`=0, `Q031`=2, `Q032`=2,
`Q033`=2, `Q041`=2, `Q042`=2, `Q043`=2) — no re-run of `run_eval.py` or
the DB was needed, only `score_layer2` for the 8 missing question ids.
Also worth knowing for next time: `adapters/llm/openai_compatible.py`'s
`_is_transient` only retries 5xx/timeout — a 429 is never retried
today, and the raised `LlmUnavailableError(str(exc))` doesn't capture
the response body, where Google puts the actual reason (quota vs.
model-retired) — a real gap if this needs diagnosing again.

Reproduce the final agreement number:

```
cd backend
.venv/Scripts/python scripts/measure_judge_agreement.py eval/reports/day18-final-human-review.csv
```

General commands for a fresh run against a different config:

```
cd backend
DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \
    .venv/Scripts/python scripts/run_eval.py --questions eval/dataset/questions.yaml
DATABASE_URL=postgresql+asyncpg://normeon:normeon@localhost:5433/normeon \
    .venv/Scripts/python scripts/score_eval_report.py eval/reports/<hash>.json
# fill in human_score in eval/reports/<hash>.human_review.csv by hand, then:
.venv/Scripts/python scripts/measure_judge_agreement.py eval/reports/<hash>.human_review.csv
```

## Retrieval and outcome metrics (Giorno 19)

**Done.** `src/app/eval/metrics.py` -- pure (no DB/LLM/network), takes
an `EvalReport` + `ScoredReport` + the question set and returns one
`EvalMetrics`. `scripts/compute_metrics.py <report>.json` reads the
matching `<hash>.scored.json` next to it and writes `<hash>.metrics.json`.

- `recall_at_k` / `mean_reciprocal_rank` / `precision_at_k`: gold-source
  matching is tolerant on purpose -- same `document`, plus *either* a
  section-string overlap *or* the page within +/-1, not an exact match
  on both, since chunk boundaries rarely land on the same split a human
  would draw by hand. Questions with no `gold_sources` (`unanswerable`)
  and errored runs are excluded from the denominator, not counted as 0.
- `citation_precision`: one global ratio (matched citations / total
  citations) across the whole run, not averaged per question -- a
  question with many citations isn't diluted to the same weight as one
  with a single lucky hit.
- `hallucinated_citation_rate` needed a real gap closed first: the
  count of invented `[S..]` markers `domain.citations.
  extract_and_validate` drops was only ever logged, never persisted on
  a `QuestionRun`. Added `invented_citations` to `AnswerResult`
  (`services/generation.py`) and threaded it through `eval/runner.py`
  into `QuestionRun` (default `0`, so old report JSON files without the
  field still load).
- `answer_accuracy` operationalizes "Layer 1 + Layer 2" as an
  escalation, not an average: Layer 1's strict substring check is
  authoritative when it passes; when it fails, Layer 2's judge score is
  consulted as a semantic fallback, and only a full score of `2` flips
  a Layer-1 failure to correct (a judge score of `1` is itself not
  fully convinced, so it doesn't count).
- `correct_abstention_rate` / `false_abstention_rate`: fraction of
  `should_abstain` / answerable questions (respectively) that actually
  abstained, among completed runs.
- `latency_ms_p50`/`p95`: linear-interpolation percentile (numpy's
  default convention), per phase and total, over answered questions
  only.
- `cost_per_query_avg`: mean of `cost_usd` where not `None` --
  `null` for a run on a model not in `pricing.yaml` (expected, not a
  bug -- see `services/pricing.calculate_cost`'s own docstring).

**Real output, run against the full 50-question set**
(`eval/reports/day18-merged-real-run.json` +
`day18-merged-real-run.scored.json`), satisfying the plan's literal
"Fatto quando" (a real JSON with every metric on a real configuration):

```json
{
  "config_hash": "day18-merged-real-run",
  "n_questions": 50,
  "n_errors": 0,
  "recall_at_k": 0.9,
  "mrr": 0.725,
  "precision_at_k": 0.2522,
  "citation_precision": 0.6591,
  "hallucinated_citation_rate": 0.0,
  "answer_accuracy": 0.7234,
  "correct_abstention_rate": 0.9,
  "false_abstention_rate": 0.2,
  "latency_ms_p50": {"retrieval_ms": 228.0, "generation_ms": 910.4, "total_ms": 1154.8},
  "latency_ms_p95": {"retrieval_ms": 4503.1, "generation_ms": 9360.1, "total_ms": 11077.8},
  "cost_per_query_avg": null
}
```

**One caveat on this specific number, not on the metric itself:** the
scored report this ran against (`day18-merged-real-run.scored.json`)
was scored on 2026-09-15, entirely before the judge-model swap to
`gemini-flash-lite-latest` (see "Judge model note" above) -- every one
of its 50 Layer 2 calls is a `JudgeError` from the 429 wall, so
`answer_accuracy` above is really Layer 1's pass rate alone, with
Layer 2's semantic fallback never engaging. Re-scoring the full 50 with
the now-working judge model would sharpen this number but costs 50 real
judge calls -- worth doing before Giorno 20's matrix run, not required
for Giorno 19's own "Fatto quando". `false_abstention_rate = 0.2` is
already a genuine, useful finding either way (independent of the judge):
8 of 40 answerable questions wrongly abstained, more than the 2 seen in
the 15-question human-review sample alone -- worth a look before
Giorno 20 tunes `min_rerank_score_for_answer`.

## First matrix run (Giorno 20)

*Not yet run.* Plan: `{fixed_500, structural} x {vector-only, hybrid}`,
reranker on, `top_k=5`. Report table + `docs/adr/0003-*.md` with the
chosen default chunking strategy, backed by these numbers.
