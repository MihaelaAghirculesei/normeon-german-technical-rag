"""Unit tests for app/eval/scoring: Layer 1 (deterministic) and Layer 2
(LLM-as-judge, against FakeLlmClient)."""

from app.adapters.llm.fake import FakeLlmClient
from app.eval.models import EvalQuestion, GoldSource
from app.eval.scoring import (
    JudgeError,
    JudgeResult,
    render_judge_prompt,
    score_layer1,
    score_layer2,
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
