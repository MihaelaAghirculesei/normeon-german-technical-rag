"""Unit tests for the OpenAI-compatible chat client, driven by an
``httpx.MockTransport`` so nothing leaves the process.
"""

import asyncio
import json
import time

import httpx
import pytest
from tenacity import AsyncRetrying, RetryCallState

from app.adapters.llm.openai_compatible import (
    MAX_RETRY_AFTER_S,
    OpenAICompatibleClient,
    _retry_after_seconds,
    _wait,
)
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


async def test_a_timeout_without_a_message_is_named_by_its_type() -> None:
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        max_retries=0,
        transport=_sequenced_transport([httpx.ReadTimeout("")]),
    )

    with pytest.raises(LlmTimeoutError, match="^ReadTimeout$"):
        await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)


# --- overall deadline ---------------------------------------------------------

#: Guards each deadline test, so a regression fails instead of hanging.
_GUARD_S = 5.0


def _stalling_transport(calls: list[httpx.Request], *, then: list | None = None):  # noqa: ANN202
    """Never answers the first request(s); answers later ones from `then`
    in order, if given -- a server that holds the connection open."""
    remaining = iter(then or [])

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        item = next(remaining, None)
        if item is None:
            await asyncio.Event().wait()
        return item

    return httpx.MockTransport(handler)


async def test_a_call_that_never_answers_raises_llm_timeout_within_the_deadline() -> None:
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        call_deadline=0.05,
        max_retries=0,
        transport=_stalling_transport([]),
    )

    started = time.monotonic()
    async with asyncio.timeout(_GUARD_S):
        with pytest.raises(LlmTimeoutError, match="0.05 s"):
            await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)
    assert time.monotonic() - started < 1.0


async def test_an_attempt_past_the_deadline_is_retried() -> None:
    calls: list[httpx.Request] = []
    stall_then_answer = _stalling_transport(
        calls, then=[None, httpx.Response(200, json=_ok_body())]
    )
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        call_deadline=0.05,
        max_retries=1,
        transport=stall_then_answer,
    )

    async with asyncio.timeout(_GUARD_S):
        resp = await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)

    assert resp.text == "Antwort [S1]."
    assert len(calls) == 2


async def test_a_slow_answer_inside_the_deadline_is_kept() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=_ok_body())

    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        call_deadline=1.0,
        max_retries=0,
        transport=httpx.MockTransport(handler),
    )

    resp = await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)

    assert resp.text == "Antwort [S1]."


def _sse_body(*chunks: dict) -> bytes:
    """Each `chunks` entry is a raw wire chunk -- typically `{"model":
    ..., "choices": [{"delta": {"content": ...}}]}` or a usage-only
    `{"choices": [], "usage": {...}}`."""
    lines = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (lines + "data: [DONE]\n\n").encode("utf-8")


def _content_chunk(text: str, *, model: str = "test-model") -> dict:
    return {"model": model, "choices": [{"delta": {"content": text}}]}


def _usage_chunk(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "choices": [],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


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
        transport=_sse_transport(_sse_body(_content_chunk("Die "), _content_chunk("Antwort."))),
    )

    deltas = [d async for d in client.stream(system="s", user="u", temperature=0.0, max_tokens=10)]

    assert "".join(d.text for d in deltas) == "Die Antwort."
    assert all(d.model == "test-model" for d in deltas)


async def test_stream_stops_at_done_and_ignores_anything_after() -> None:
    after_done = json.dumps(_content_chunk("spaeter"))
    body = _sse_body(_content_chunk("Die ")) + f"data: {after_done}\n\n".encode()
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1", api_key=None, model="m", transport=_sse_transport(body)
    )

    deltas = [d async for d in client.stream(system="s", user="u", temperature=0.0, max_tokens=10)]

    assert "".join(d.text for d in deltas) == "Die "


async def test_stream_skips_deltas_with_no_content_and_no_usage() -> None:
    role_only_chunk = {"model": "m", "choices": [{"delta": {"role": "assistant"}}]}
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        transport=_sse_transport(_sse_body(role_only_chunk, _content_chunk("Text"))),
    )

    deltas = [d async for d in client.stream(system="s", user="u", temperature=0.0, max_tokens=10)]

    assert [d.text for d in deltas] == ["Text"]


async def test_stream_yields_a_final_usage_only_delta() -> None:
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        transport=_sse_transport(_sse_body(_content_chunk("Text"), _usage_chunk(42, 7))),
    )

    deltas = [d async for d in client.stream(system="s", user="u", temperature=0.0, max_tokens=10)]

    assert [d.text for d in deltas] == ["Text", ""]
    assert deltas[0].prompt_tokens is None and deltas[0].completion_tokens is None
    assert (deltas[1].prompt_tokens, deltas[1].completion_tokens) == (42, 7)


async def test_stream_sends_stream_true_and_include_usage() -> None:
    transport = _sse_transport(_sse_body(_content_chunk("x")))
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1", api_key=None, model="m", transport=transport
    )

    async for _ in client.stream(system="s", user="u", temperature=0.0, max_tokens=10):
        pass

    sent = json.loads(transport.captured[0].content)  # type: ignore[attr-defined]
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


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


# --- stream deadlines ---------------------------------------------------------


def _paced_sse_transport(pieces: list[bytes | None], *, gap: float) -> httpx.MockTransport:
    """Sends each piece `gap` seconds after the previous one; a `None`
    piece stalls the body forever -- a server gone quiet mid-answer."""

    async def body():  # noqa: ANN202
        for piece in pieces:
            await asyncio.sleep(gap)
            if piece is None:
                await asyncio.Event().wait()
            else:
                yield piece

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


def _sse_line(chunk: dict) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _deadline_client(transport: httpx.MockTransport) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        first_token_timeout=0.1,
        stream_idle_timeout=0.1,
        transport=transport,
    )


async def _drain(client: OpenAICompatibleClient, into: list[str]) -> None:
    async for delta in client.stream(system="s", user="u", temperature=0.0, max_tokens=10):
        into.append(delta.text)


async def test_stream_that_never_answers_raises_llm_timeout() -> None:
    client = _deadline_client(_stalling_transport([]))

    async with asyncio.timeout(_GUARD_S):
        with pytest.raises(LlmTimeoutError, match="no first token within 0.1 s"):
            await _drain(client, [])


async def test_stream_whose_body_never_starts_raises_llm_timeout() -> None:
    client = _deadline_client(_paced_sse_transport([None], gap=0.0))

    async with asyncio.timeout(_GUARD_S):
        with pytest.raises(LlmTimeoutError, match="no first token"):
            await _drain(client, [])


async def test_stream_that_stalls_mid_answer_keeps_what_arrived_then_times_out() -> None:
    received: list[str] = []
    client = _deadline_client(
        _paced_sse_transport([_sse_line(_content_chunk("Die ")), None], gap=0.0)
    )

    async with asyncio.timeout(_GUARD_S):
        with pytest.raises(LlmTimeoutError, match="no new token within 0.1 s"):
            await _drain(client, received)
    assert received == ["Die "]


async def test_stream_keep_alives_do_not_count_as_progress() -> None:
    client = _deadline_client(_paced_sse_transport([b": keep-alive\n\n"] * 1000, gap=0.02))

    async with asyncio.timeout(_GUARD_S):
        with pytest.raises(LlmTimeoutError, match="no first token"):
            await _drain(client, [])


async def test_stream_slow_but_steady_is_not_cut() -> None:
    """Twelve deltas 0.05 s apart take longer in total than either
    timeout (0.1 s); every gap is inside the idle limit, so all arrive."""
    words = [f"w{i} " for i in range(12)]
    pieces: list[bytes | None] = [_sse_line(_content_chunk(w)) for w in words]
    client = _deadline_client(_paced_sse_transport([*pieces, b"data: [DONE]\n\n"], gap=0.05))
    received: list[str] = []

    async with asyncio.timeout(_GUARD_S):
        await _drain(client, received)
    assert received == words


# --- rate limiting (429) ------------------------------------------------------


async def test_a_429_is_retried_and_can_still_succeed() -> None:
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        max_retries=1,
        transport=_sequenced_transport(
            [
                httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "slow down"}),
                httpx.Response(200, json=_ok_body()),
            ]
        ),
    )

    resp = await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)

    assert resp.text == "Antwort [S1]."


async def test_a_429_exhausting_retries_raises_llm_unavailable() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "quota"})

    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key=None,
        model="m",
        max_retries=2,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LlmUnavailableError):
        await client.complete(system="s", user="u", temperature=0.0, max_tokens=10)
    assert len(captured) == 3


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("7", 7.0),
        ("0.5", 0.5),
        (None, None),
        ("soon", None),
        ("-1", None),
        ("Wed, 21 Oct 2026 07:28:00 GMT", None),
    ],
)
def test_retry_after_parsing(header: str | None, expected: float | None) -> None:
    assert _retry_after_seconds(header) == expected


def _state_after(response: httpx.Response) -> RetryCallState:
    state = RetryCallState(retry_object=AsyncRetrying(), fn=None, args=(), kwargs={})
    state.attempt_number = 1
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    error = httpx.HTTPStatusError("x", request=request, response=response)
    state.set_exception((type(error), error, None))
    return state


def test_wait_honours_retry_after_up_to_the_cap() -> None:
    assert _wait(_state_after(httpx.Response(429, headers={"Retry-After": "3"}))) == 3.0
    assert _wait(_state_after(httpx.Response(429, headers={"Retry-After": "600"}))) == (
        MAX_RETRY_AFTER_S
    )


def test_wait_falls_back_to_backoff_without_retry_after() -> None:
    assert _wait(_state_after(httpx.Response(429))) <= 4
    assert _wait(_state_after(httpx.Response(503, headers={"Retry-After": "600"}))) <= 4
