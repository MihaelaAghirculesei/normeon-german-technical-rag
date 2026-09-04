"""Unit tests for the versioned-prompt loader.

These read the real shipped prompt file -- it is part of the contract
(its name and hash are logged, its instructions drive every answer), so
a change to it should break a test on purpose.
"""

import hashlib
from pathlib import Path

import pytest

from app.services.prompts import (
    CONTEXT_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
    load_prompt,
)

PROMPTS_DIR = Path(__file__).parents[2] / "prompts"


def test_load_prompt_returns_name_text_and_matching_sha256() -> None:
    prompt = load_prompt("answer_de.v1")
    raw = (PROMPTS_DIR / "answer_de.v1.txt").read_bytes()

    assert prompt.name == "answer_de.v1"
    assert prompt.template == raw.decode("utf-8")
    assert prompt.sha256 == hashlib.sha256(raw).hexdigest()
    assert len(prompt.sha256) == 64


def test_load_prompt_is_cached() -> None:
    assert load_prompt("answer_de.v1") is load_prompt("answer_de.v1")


def test_unknown_prompt_name_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")


def test_render_fills_both_placeholders_and_leaves_the_rest() -> None:
    prompt = load_prompt("answer_de.v1")

    rendered = prompt.render(context="[S1] ...", question="Welche Lenkkraft?")

    assert CONTEXT_PLACEHOLDER not in rendered
    assert QUESTION_PLACEHOLDER not in rendered
    assert "[S1] ..." in rendered
    assert "Welche Lenkkraft?" in rendered
    assert "Antworte auf Deutsch." in rendered


def test_render_does_not_choke_on_braces_or_percent_in_the_text() -> None:
    prompt = load_prompt("answer_de.v1")

    rendered = prompt.render(
        context="[S1] Formel: a{b} bei 50 % Last", question="und {jetzt}?"
    )

    assert "a{b} bei 50 % Last" in rendered
    assert "und {jetzt}?" in rendered


def test_shipped_prompt_states_the_citation_and_abstention_rules() -> None:
    template = load_prompt("answer_de.v1").template

    assert CONTEXT_PLACEHOLDER in template
    assert QUESTION_PLACEHOLDER in template
    assert "[S1]" in template
    assert "NICHT_GEFUNDEN" in template
