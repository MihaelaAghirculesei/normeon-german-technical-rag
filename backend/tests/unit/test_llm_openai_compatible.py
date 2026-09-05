"""Unit tests for the OpenAI-compatible chat client, driven by an
``httpx.MockTransport`` so nothing leaves the process.
"""

import json

import httpx
import pytest

from app.adapters.llm.openai_compatible import OpenAICompatibleClient
from app.core.errors import LlmTimeoutError, LlmUnavailableError


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


def _sequenced_transport(responses: list) -> httpx.MockTransport:  # noqa: ANN001
    """Returns (or raises) each entry in `responses` in order, one per
    request -- for exercising a transient failure followed by success."""
    remaining = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        item = next(remaining)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.MockTransport(handler)


async def test_a_5xx_is_retried_and_can_still_succeed() -> None:
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        max_retries=1,
        transport=_sequenced_transport(
            [httpx.Response(503, json={"error": "busy"}), httpx.Response(200, json=_ok_body())]
        ),
    )

    resp = await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)

    assert resp.text == "Antwort [S1]."


async def test_a_5xx_exhausting_retries_raises_llm_unavailable() -> None:
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        max_retries=1,
        transport=_transport([], body={"error": "boom"}, status=500),
    )

    with pytest.raises(LlmUnavailableError) as exc_info:
        await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)
    assert exc_info.value.code == "llm_unavailable"


async def test_a_4xx_is_not_retried() -> None:
    captured: list[httpx.Request] = []
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        max_retries=2,
        transport=_transport(captured, body={"error": "bad request"}, status=400),
    )

    with pytest.raises(LlmUnavailableError):
        await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)
    assert len(captured) == 1, "a 4xx must fail on the first attempt, never retried"


async def test_a_timeout_raises_llm_timeout() -> None:
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        max_retries=0,
        transport=_sequenced_transport([httpx.ConnectTimeout("boom")]),
    )

    with pytest.raises(LlmTimeoutError) as exc_info:
        await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)
    assert exc_info.value.code == "llm_timeout"


def _sse_body(*deltas: dict) -> bytes:
    lines = "".join(f"data: {json.dumps({'choices': [{'delta': d}]})}\n\n" for d in deltas)
    return (lines + "data: [DONE]\n\n").encode("utf-8")


def _sse_transport(body: bytes, *, status: int = 200) -> httpx.MockTransport:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status, content=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    transport.captured = captured  # type: ignore[attr-defined]
    return transport


async def test_stream_yields_delta_content_pieces_in_order() -> None:
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        transport=_sse_transport(_sse_body({"content": "Die "}, {"content": "Antwort."})),
    )

    chunks = [
        c async for c in client.stream(system="s", user="u", temperature=0.0, max_tokens=10)
    ]

    assert "".join(chunks) == "Die Antwort."


async def test_stream_stops_at_done_and_ignores_anything_after() -> None:
    after_done = json.dumps({"choices": [{"delta": {"content": "spaeter"}}]})
    body = _sse_body({"content": "Die "}) + f"data: {after_done}\n\n".encode()
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1", api_key=None, model="m", transport=_sse_transport(body)
    )

    chunks = [
        c async for c in client.stream(system="s", user="u", temperature=0.0, max_tokens=10)
    ]

    assert "".join(chunks) == "Die "


async def test_stream_skips_deltas_with_no_content() -> None:
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        transport=_sse_transport(_sse_body({"role": "assistant"}, {"content": "Text"})),
    )

    chunks = [
        c async for c in client.stream(system="s", user="u", temperature=0.0, max_tokens=10)
    ]

    assert chunks == ["Text"]


async def test_stream_sends_stream_true_in_the_payload() -> None:
    transport = _sse_transport(_sse_body({"content": "x"}))
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1", api_key=None, model="m", transport=transport
    )

    async for _ in client.stream(system="s", user="u", temperature=0.0, max_tokens=10):
        pass

    sent = json.loads(transport.captured[0].content)  # type: ignore[attr-defined]
    assert sent["stream"] is True


async def test_stream_raises_llm_unavailable_on_a_5xx() -> None:
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        transport=_sse_transport(b"", status=500),
    )

    with pytest.raises(LlmUnavailableError):
        async for _ in client.stream(system="s", user="u", temperature=0.0, max_tokens=10):
            pass


async def test_stream_raises_llm_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom")

    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LlmTimeoutError):
        async for _ in client.stream(system="s", user="u", temperature=0.0, max_tokens=10):
            pass
