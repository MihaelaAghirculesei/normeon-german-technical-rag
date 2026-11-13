"""Sweep the pre-generation confidence gate over finished cross-encoder
eval runs, without a single LLM call.

    cd backend
    .venv/Scripts/python scripts/sweep_confidence_gate.py \
        --reports-dir eval/reports/matrix <hash> [<hash> ...]

The reports do not store scores, so each question's top reranked score
is read back from the matrix's rerank cache (`rerank_cache.sqlite3` in
the reports directory): the key is the question and the top chunk's
content, which comes from the dev database (DATABASE_URL). Only runs
reranked by the cross-encoder qualify -- see `app.eval.gate`.
"""

import argparse
import asyncio
import hashlib
import sqlite3
import sys
from pathlib import Path

from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine
from app.eval.dataset import DATASET_DIR, load_questions
from app.eval.gate import GateQuestion, separability_auc, simulate_gate
from app.eval.metrics import is_correct
from app.eval.models import EvalReport
from app.eval.scoring import ScoredReport

THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


async def _contents(chunk_ids: list[str]) -> dict[str, str]:
    try:
        async with engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT id::text, content FROM chunks WHERE id::text = ANY(:ids)"),
                {"ids": chunk_ids},
            )
            return {chunk_id: content for chunk_id, content in rows}
    finally:
        # One asyncio.run per report: pooled connections must not outlive
        # the loop that opened them.
        await engine.dispose()


def _cache_key(namespace: str, query: str, content: str) -> str:
    # Must match app.adapters.reranker.caching.CachingReranker._key.
    return hashlib.sha256("\0".join((namespace, query, content)).encode("utf-8")).hexdigest()


def _load_run(
    reports_dir: Path, config_hash: str, questions_path: Path, namespace: str
) -> tuple[EvalReport, list[GateQuestion]]:
    report = EvalReport.model_validate_json(
        (reports_dir / f"{config_hash}.json").read_text(encoding="utf-8")
    )
    if report.config.reranker != "cross_encoder":
        raise SystemExit(f"{config_hash}: reranker {report.config.reranker!r}, not cross_encoder")
    scored = ScoredReport.model_validate_json(
        (reports_dir / f"{config_hash}.scored.json").read_text(encoding="utf-8")
    )
    by_id = {q.id: q for q in load_questions(questions_path)}
    scored_by_id = {s.question_id: s for s in scored.results}

    top_ids = [str(r.retrieved[0].chunk_id) for r in report.results if r.retrieved]
    contents = asyncio.run(_contents(top_ids))
    cache = sqlite3.connect(reports_dir / "rerank_cache.sqlite3")
    try:
        rows = []
        for run in report.results:
            question = by_id[run.question_id]
            top: float | None = None
            if run.retrieved:
                content = contents[str(run.retrieved[0].chunk_id)]
                hit = cache.execute(
                    "SELECT score FROM rerank_scores WHERE key = ?",
                    (_cache_key(namespace, question.question, content),),
                ).fetchone()
                if hit is None:
                    raise SystemExit(f"{config_hash} {run.question_id}: score not in the cache")
                top = float(hit[0])
            rows.append(
                GateQuestion(
                    question_id=run.question_id,
                    should_abstain=question.should_abstain,
                    top_score=top,
                    abstained=run.abstained,
                    correct=is_correct(scored_by_id[run.question_id]),
                )
            )
    finally:
        cache.close()
    return report, rows


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hashes", nargs="+", help="config hashes of cross-encoder runs")
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--questions", default=str(DATASET_DIR / "questions.yaml"))
    parser.add_argument(
        "--max-length",
        type=int,
        default=settings.reranker_max_length,
        help="the reranker token cap the runs used (part of the cache key)",
    )
    args = parser.parse_args()
    namespace = f"{settings.reranker_model}|max_length={args.max_length}"

    for config_hash in args.hashes:
        report, rows = _load_run(
            Path(args.reports_dir), config_hash, Path(args.questions), namespace
        )
        lowest_correct = min(
            (q.top_score for q in rows if q.correct and not q.should_abstain and q.top_score),
            default=None,
        )
        cfg = report.config
        print(
            f"\n### {config_hash[:12]} -- {cfg.chunking_strategy}/{cfg.retrieval_mode}, "
            f"top_k {cfg.top_k}, {cfg.embedding_provider}\n"
        )
        print(
            f"AUC answerable vs unanswerable: {separability_auc(rows):.2f}; "
            f"lowest top score of a correct answer: {lowest_correct:.3f}\n"
        )
        print(
            "| threshold | answer_accuracy | false_abstention | correct_abstention "
            "| LLM calls saved |"
        )
        print("| --- | --- | --- | --- | --- |")
        for threshold in THRESHOLDS:
            p = simulate_gate(rows, threshold)
            print(
                f"| {threshold:.1f} | {p.answer_accuracy:.3f} | {p.false_abstention_rate:.3f} "
                f"| {p.correct_abstention_rate:.3f} | {p.llm_calls_saved} |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
