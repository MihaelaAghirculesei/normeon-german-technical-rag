"""Unit tests for the OpenAI-compatible chat client, driven by an
``httpx.MockTransport`` so nothing leaves the process.
"""

import json

import httpx
import pytest

from app.adapters.llm.openai_compatible import OpenAICompatibleClient


def _transport(
    captured: list[httpx.Request], *, body: dict, status: int = 200
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


def _ok_body(content: str = "Antwort [S1].") -> dict:
    return {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7},
    }


async def test_posts_chat_completions_with_the_expected_payload() -> None:
    captured: list[httpx.Request] = []
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key="secret",
        model="test-model",
        transport=_transport(captured, body=_ok_body()),
    )

    await client.complete(system="SYS", user="USR", temperature=0.0, max_tokens=800)

    (request,) = captured
    assert request.method == "POST"
    assert str(request.url) == "https://llm.example/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    sent = json.loads(request.content)
    assert sent["model"] == "test-model"
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 800
    assert sent["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]


async def test_parses_content_model_and_usage() -> None:
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="fallback",
        transport=_transport([], body=_ok_body(content="  Antwort [S1].  ")),
    )

    resp = await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)

    assert resp.text == "Antwort [S1]."
    assert resp.model == "test-model"
    assert (resp.prompt_tokens, resp.completion_tokens) == (42, 7)


async def test_missing_usage_block_yields_none_token_counts() -> None:
    body = {"choices": [{"message": {"content": "x"}}]}
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="fallback",
        transport=_transport([], body=body),
    )

    resp = await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)

    assert resp.model == "fallback"
    assert resp.prompt_tokens is None and resp.completion_tokens is None


async def test_no_api_key_sends_no_authorization_header() -> None:
    captured: list[httpx.Request] = []
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        transport=_transport(captured, body=_ok_body()),
    )

    await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)

    assert "authorization" not in {k.lower() for k in captured[0].headers}


async def test_http_error_propagates() -> None:
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        transport=_transport([], body={"error": "boom"}, status=500),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)
