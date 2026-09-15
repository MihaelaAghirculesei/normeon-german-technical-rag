"""Unit tests for app/eval/scoring: Layer 1 (deterministic) and Layer 2
(LLM-as-judge, against FakeLlmClient)."""

from datetime import UTC, datetime
from pathlib import Path

from app.adapters.llm.fake import FakeLlmClient
from app.eval.models import EvalConfig, EvalQuestion, EvalReport, GoldSource, QuestionRun
from app.eval.scoring import (
    JudgeError,
    JudgeResult,
    ScoredReport,
    render_judge_prompt,
    score_layer1,
    score_layer2,
    score_report,
    write_scored_report,
)


def _question(**overrides: object) -> EvalQuestion:
    defaults: dict[str, object] = {
        "id": "Q001",
        "category": "requirement_lookup",
        "question": "Welche Lenkkraft ist zulaessig?",
        "expected_answer_points": ["300 N"],
        "gold_sources": [GoldSource(document="StVZO.pdf", section="§41", page=40)],
        "should_abstain": False,
    }
    defaults.update(overrides)
    return EvalQuestion.model_validate(defaults)


def _question_run(question_id: str, answer: str) -> QuestionRun:
    return QuestionRun(
        question_id=question_id,
        category="requirement_lookup",
        answer=answer,
        abstained=False,
        citations=[],
        retrieved=[],
        retrieval_ms=1.0,
        generation_ms=1.0,
        total_ms=2.0,
        cost_usd=None,
        prompt_name="answer_de.v1",
        prompt_sha256="deadbeef",
        model="fake",
    )


def _report(*runs: QuestionRun) -> EvalReport:
    return EvalReport(
        config=EvalConfig(
            chunking_strategy="structural",
            reranker="noop",
            embedding_provider="fake",
            llm_provider="fake",
            llm_model="fake",
        ),
        config_hash="testhash",
        created_at=datetime.now(UTC),
        n_questions=len(runs),
        results=list(runs),
    )


def test_layer1_finds_an_exact_point() -> None:
    result = score_layer1(["300 N"], "Die zulaessige Lenkkraft betraegt 300 N.")

    assert result.points_found == ["300 N"]
    assert result.all_present is True


def test_layer1_tolerates_decimal_comma_vs_dot() -> None:
    result = score_layer1(["5,0 m/s2"], "Es sind mindestens 5.0 m/s2 vorgeschrieben.")

    assert result.all_present is True


def test_layer1_tolerates_case_and_whitespace() -> None:
    result = score_layer1(["ip6k9k"], "Schutzart:   IP6K9K")

    assert result.all_present is True


def test_layer1_reports_missing_points_in_original_order() -> None:
    result = score_layer1(["300 N", "250 N"], "Die Lenkkraft betraegt 300 N.")

    assert result.points_found == ["300 N"]
    assert result.points_missing == ["250 N"]
    assert result.all_present is False


def test_layer1_empty_points_is_trivially_satisfied() -> None:
    result = score_layer1([], "Beliebiger Text.")

    assert result.all_present is True


def test_render_judge_prompt_fills_every_placeholder() -> None:
    question = _question(should_abstain=True, expected_answer_points=[])

    rendered = render_judge_prompt(question, "NICHT_GEFUNDEN")

    assert "{{" not in rendered
    assert "Welche Lenkkraft ist zulaessig?" in rendered
    assert "ja" in rendered  # should_abstain
    assert "StVZO.pdf" in rendered
    assert "NICHT_GEFUNDEN" in rendered


async def test_score_layer2_parses_a_well_formed_judge_response() -> None:
    llm = FakeLlmClient(canned='{"score": 2, "rationale": "Alle Fakten korrekt genannt."}')

    result = await score_layer2(llm, question=_question(), answer="Die Lenkkraft betraegt 300 N.")

    assert isinstance(result, JudgeResult)
    assert result.score == 2
    assert result.rationale == "Alle Fakten korrekt genannt."


async def test_score_layer2_surfaces_malformed_json_as_judge_error() -> None:
    llm = FakeLlmClient(canned="das ist kein JSON")

    result = await score_layer2(llm, question=_question(), answer="irgendeine Antwort")

    assert isinstance(result, JudgeError)
    assert result.raw_response == "das ist kein JSON"


async def test_score_layer2_surfaces_out_of_range_score_as_judge_error() -> None:
    llm = FakeLlmClient(canned='{"score": 5, "rationale": "zu hoch"}')

    result = await score_layer2(llm, question=_question(), answer="irgendeine Antwort")

    assert isinstance(result, JudgeError)


async def test_score_report_scores_every_matching_run() -> None:
    questions = [_question(id="Q001"), _question(id="Q002", expected_answer_points=["250 N"])]
    report = _report(
        _question_run("Q001", "Die Lenkkraft betraegt 300 N."),
        _question_run("Q002", "Die Lenkkraft betraegt 300 N."),  # wrong value on purpose
    )
    llm = FakeLlmClient(canned='{"score": 1, "rationale": "teilweise korrekt"}')

    scored = await score_report(report, questions, llm)

    assert scored.config_hash == "testhash"
    assert [r.question_id for r in scored.results] == ["Q001", "Q002"]
    assert scored.results[0].layer1.all_present is True
    assert scored.results[1].layer1.all_present is False
    assert all(isinstance(r.layer2, JudgeResult) and r.layer2.score == 1 for r in scored.results)


async def test_score_report_skips_a_run_whose_question_no_longer_exists() -> None:
    questions = [_question(id="Q001")]
    report = _report(_question_run("Q001", "300 N"), _question_run("Q999", "irrelevant"))
    llm = FakeLlmClient(canned='{"score": 2, "rationale": "ok"}')

    scored = await score_report(report, questions, llm)

    assert [r.question_id for r in scored.results] == ["Q001"]


def test_write_scored_report_round_trips(tmp_path: Path) -> None:
    scored = ScoredReport(config_hash="testhash", scored_at=datetime.now(UTC), results=[])

    path = write_scored_report(scored, reports_dir=tmp_path)

    assert path == tmp_path / "testhash.scored.json"
    assert ScoredReport.model_validate_json(path.read_text(encoding="utf-8")) == scored
