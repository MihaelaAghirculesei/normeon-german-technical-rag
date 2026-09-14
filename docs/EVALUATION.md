# Evaluation

Status: work in progress (plan, Giorni 17-20). This document tracks the
50-question evaluation set, the two-layer correctness metric, and the
judge-agreement check. Sections below are filled in as each part lands.

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
lands in `tests/unit/` once the set is complete.

## Correctness metric (Giorno 18)

*Not yet implemented.*

- Layer 1 — deterministic: `expected_answer_points` present in the answer
  (normalized match, tolerant of numeric formatting).
- Layer 2 — LLM-as-judge: 0-2 scale, given the answer, the expected
  points, and the gold sources.

## Judge agreement (Giorno 18)

*Not yet measured.* Plan: score 15 answers by hand, compare against the
judge's scores, report percent agreement or Cohen's kappa here.

## Retrieval and outcome metrics (Giorno 19)

*Not yet implemented.* See `src/app/eval/metrics.py` (planned):
`recall@k`, `mrr`, `precision@k`, `citation_precision`,
`hallucinated_citation_rate`, `answer_accuracy`, `correct_abstention_rate`,
`false_abstention_rate`, `latency_p50`/`p95`, `cost_per_query_avg`.

## First matrix run (Giorno 20)

*Not yet run.* Plan: `{fixed_500, structural} x {vector-only, hybrid}`,
reranker on, `top_k=5`. Report table + `docs/adr/0003-*.md` with the
chosen default chunking strategy, backed by these numbers.
