"""For every false abstention of finished eval runs, show where each
expected answer point was: in the context the model saw, only in a
neighbouring chunk (`ordinal` +-1, same document and chunking), or in
neither -- the upper bound on what adding neighbours to the context
could fix, without a single LLM call.

    cd backend
    .venv/Scripts/python scripts/check_neighbour_coverage.py \
        --reports-dir eval/reports/matrix <hash> [<hash> ...]

Reads chunk texts from the dev database (DATABASE_URL).
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

from app.db.session import engine
from app.eval.coverage import PointLocation, locate_point, neighbours_could_answer
from app.eval.dataset import DATASET_DIR, load_questions
from app.eval.models import EvalReport

_CONTEXT_SQL = text("SELECT content FROM chunks WHERE id::text = ANY(:ids)")
_NEIGHBOURS_SQL = text(
    """
    SELECT DISTINCT n.id, n.content
    FROM chunks c
    JOIN chunks n
      ON n.document_id = c.document_id
     AND n.strategy = c.strategy
     AND n.ordinal BETWEEN c.ordinal - 1 AND c.ordinal + 1
    WHERE c.id::text = ANY(:ids) AND NOT n.id::text = ANY(:ids)
    """
)


async def _report(report: EvalReport, questions_path: Path) -> None:
    by_id = {q.id: q for q in load_questions(questions_path)}
    cfg = report.config
    print(
        f"\n### {report.config_hash[:12]} -- {cfg.chunking_strategy}/{cfg.retrieval_mode}, "
        f"{cfg.reranker}, {cfg.embedding_provider}\n"
    )
    print("| question | category | answer points |")
    print("| --- | --- | --- |")
    n_false = n_fixable = 0
    totals: dict[PointLocation, int] = {"context": 0, "neighbour": 0, "missing": 0}
    async with engine.connect() as conn:
        for run in report.results:
            question = by_id[run.question_id]
            if question.should_abstain or not run.abstained or run.error is not None:
                continue
            n_false += 1
            ids = [str(r.chunk_id) for r in run.retrieved]
            context = " ".join((await conn.scalars(_CONTEXT_SQL, {"ids": ids})).all())
            neighbours = " ".join(
                content for _, content in (await conn.execute(_NEIGHBOURS_SQL, {"ids": ids})).all()
            )
            locations = [
                locate_point(p, context, neighbours) for p in question.expected_answer_points
            ]
            for location in locations:
                totals[location] += 1
            n_fixable += neighbours_could_answer(locations)
            cells = ", ".join(
                f"{p} ({loc})"
                for p, loc in zip(question.expected_answer_points, locations, strict=True)
            )
            print(f"| {run.question_id} | {question.category} | {cells} |")
    print(
        f"\n{n_false} false abstentions; answer points: {totals}; "
        f"neighbours could complete the context for {n_fixable} of "
        f"{len(report.results)} questions."
    )


async def _main(args: argparse.Namespace) -> None:
    try:
        for config_hash in args.hashes:
            path = Path(args.reports_dir) / f"{config_hash}.json"
            report = EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
            await _report(report, Path(args.questions))
    finally:
        await engine.dispose()


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hashes", nargs="+", help="config hashes of finished runs")
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--questions", default=str(DATASET_DIR / "questions.yaml"))
    asyncio.run(_main(parser.parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
