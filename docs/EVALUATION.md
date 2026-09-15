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

**Code done, not yet run for real** — needs an actual (non-`fake`) LLM
provider configured; see "Blocked on" below.

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

**Code done, not yet measured** — needs the actual run above first.

- `select_human_review_sample` (`src/app/eval/agreement.py`) picks 3
  questions per category (15 total, deterministic — the first 3 by id
  in each category), so every category is represented.
- `write_human_review_worksheet` exports those 15 as a CSV: question,
  the system's answer, the gold facts/sources, the judge's own score —
  and a blank `human_score` column.
- Fill in `human_score` by hand, then run
  `scripts/measure_judge_agreement.py <worksheet.csv>` — prints percent
  agreement and Cohen's kappa (`cohens_kappa`/`percent_agreement` in
  `agreement.py`, unit-tested against a hand-worked confusion matrix).
  Paste the result here once run.

### Blocked on: a real LLM provider

`core/config.py`'s `llm_provider` defaults to `"fake"` (`FakeLlmClient`,
a deterministic canned responder used everywhere else in this project's
test suite) and no API key is configured anywhere in this repo (no
`.env`, nothing in `docker-compose.yml`). Running the 50 questions for
real answers, and judging them for real, needs `LLM_PROVIDER=
openai_compatible` + `LLM_API_BASE_URL` + `LLM_API_KEY` + `LLM_MODEL`
pointed at an actual provider (e.g. OpenRouter, per `backend/pricing.
yaml`'s existing model list) — a cost/provider decision for the user,
not something to default silently. Once that's set:

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

*Not yet implemented.* See `src/app/eval/metrics.py` (planned):
`recall@k`, `mrr`, `precision@k`, `citation_precision`,
`hallucinated_citation_rate`, `answer_accuracy`, `correct_abstention_rate`,
`false_abstention_rate`, `latency_p50`/`p95`, `cost_per_query_avg`.

## First matrix run (Giorno 20)

*Not yet run.* Plan: `{fixed_500, structural} x {vector-only, hybrid}`,
reranker on, `top_k=5`. Report table + `docs/adr/0003-*.md` with the
chosen default chunking strategy, backed by these numbers.
