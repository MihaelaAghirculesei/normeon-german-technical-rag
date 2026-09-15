"""Two-layer correctness metric (plan, Giorno 18).

Layer 1 -- deterministic: every `expected_answer_points` string must
appear in the answer after a normalization that only smooths numeric
*formatting* (decimal comma vs. dot, whitespace, case) -- not a fuzzy or
semantic match. Zero cost, zero LLM calls, fully reproducible: the
honest baseline Layer 2 gets measured against.

Layer 2 -- LLM-as-judge: a versioned prompt (`prompts/eval_judge_de.v1.
txt`) scores 0-2 given the question, the expected points, the gold
sources, and the answer. Reuses `services.prompts.load_prompt` for the
read+hash (so the judge prompt's sha256 is as reproducible as the answer
prompt's) but NOT `Prompt.render` -- that method's placeholder contract
(`{{CONTEXT}}`/`{{QUESTION}}`) is the answer prompt's own, not a generic
templating facility, and the judge prompt has different placeholders.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from app.adapters.llm.base import LlmClient
from app.eval.models import Category, EvalQuestion, EvalReport
from app.eval.runner import REPORTS_DIR
from app.services.prompts import load_prompt

JUDGE_PROMPT_NAME = "eval_judge_de.v1"

# A short, fixed steer, same split as services/generation.py's own
# _SYSTEM: the behavioural contract lives in the versioned prompt file.
_SYSTEM = (
    "Du bist ein strenger, praeziser Gutachter fuer Antworten eines "
    "technischen RAG-Systems. Halte dich strikt an die folgenden Anweisungen."
)


class Layer1Result(BaseModel):
    points_found: list[str]
    points_missing: list[str]

    @property
    def all_present(self) -> bool:
        return not self.points_missing


class JudgeResult(BaseModel):
    score: Literal[0, 1, 2]
    rationale: str


class JudgeError(BaseModel):
    """The judge's response didn't parse as the expected JSON shape --
    surfaced as data, not raised, so one malformed judge call doesn't
    abort scoring a whole eval run."""

    raw_response: str
    error: str


def _normalize(text: str) -> str:
    normalized = text.lower().replace(" ", " ")
    normalized = re.sub(r"(?<=\d),(?=\d)", ".", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def score_layer1(expected_answer_points: list[str], answer: str) -> Layer1Result:
    normalized_answer = _normalize(answer)
    found = [p for p in expected_answer_points if _normalize(p) in normalized_answer]
    found_set = set(found)
    missing = [p for p in expected_answer_points if p not in found_set]
    return Layer1Result(points_found=found, points_missing=missing)


def _format_points(points: list[str]) -> str:
    return "\n".join(f"- {p}" for p in points) if points else "(keine)"


def _format_gold_sources(question: EvalQuestion) -> str:
    if not question.gold_sources:
        return "(keine)"
    lines = []
    for source in question.gold_sources:
        location = ", ".join(
            filter(None, [source.section, f"S. {source.page}" if source.page else None])
        )
        lines.append(f"- {source.document}" + (f" ({location})" if location else ""))
    return "\n".join(lines)


def render_judge_prompt(question: EvalQuestion, answer: str) -> str:
    template = load_prompt(JUDGE_PROMPT_NAME).template
    return (
        template.replace("{{QUESTION}}", question.question)
        .replace("{{SHOULD_ABSTAIN}}", "ja" if question.should_abstain else "nein")
        .replace("{{EXPECTED_POINTS}}", _format_points(question.expected_answer_points))
        .replace("{{GOLD_SOURCES}}", _format_gold_sources(question))
        .replace("{{ANSWER}}", answer)
    )


def _parse_judge_response(text: str) -> JudgeResult | JudgeError:
    try:
        data = json.loads(text)
        return JudgeResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        return JudgeError(raw_response=text, error=str(exc))


async def score_layer2(
    llm: LlmClient,
    *,
    question: EvalQuestion,
    answer: str,
    temperature: float = 0.0,
    max_tokens: int = 300,
) -> JudgeResult | JudgeError:
    user = render_judge_prompt(question, answer)
    response = await llm.complete(
        system=_SYSTEM, user=user, temperature=temperature, max_tokens=max_tokens
    )
    return _parse_judge_response(response.text)


class ScoredQuestionRun(BaseModel):
    question_id: str
    category: Category
    answer: str
    layer1: Layer1Result
    layer2: JudgeResult | JudgeError


class ScoredReport(BaseModel):
    config_hash: str
    scored_at: datetime
    results: list[ScoredQuestionRun]


async def score_report(
    report: EvalReport, questions: list[EvalQuestion], llm: LlmClient
) -> ScoredReport:
    """Layer 1 + Layer 2 for every answered question in `report`. A
    `QuestionRun` whose `question_id` no longer matches any question in
    `questions` (the set changed since the report was run) is skipped
    rather than raising -- a stale report is a caller's mistake to
    notice from a `ScoredReport` shorter than `report.results`, not this
    function's to guess around."""
    by_id = {q.id: q for q in questions}
    scored: list[ScoredQuestionRun] = []
    for run in report.results:
        question = by_id.get(run.question_id)
        if question is None:
            continue
        layer1 = score_layer1(question.expected_answer_points, run.answer)
        layer2 = await score_layer2(llm, question=question, answer=run.answer)
        scored.append(
            ScoredQuestionRun(
                question_id=run.question_id,
                category=question.category,
                answer=run.answer,
                layer1=layer1,
                layer2=layer2,
            )
        )
    return ScoredReport(
        config_hash=report.config_hash, scored_at=datetime.now(UTC), results=scored
    )


def write_scored_report(scored: ScoredReport, reports_dir: Path = REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{scored.config_hash}.scored.json"
    path.write_text(scored.model_dump_json(indent=2), encoding="utf-8")
    return path
