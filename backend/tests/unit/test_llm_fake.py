"""Unit tests for the deterministic FakeLlmClient (asyncio_mode = auto)."""

from app.adapters.llm.fake import FakeLlmClient


async def test_cites_the_markers_present_in_the_user_message() -> None:
    client = FakeLlmClient()

    resp = await client.complete(
        system="x",
        user="[S1] Quelle: a.pdf\n...\n\n[S2] Quelle: b.pdf\n...",
        temperature=0.0,
        max_tokens=100,
    )

    assert resp.model == "fake"
    assert "[S1]" in resp.text and "[S2]" in resp.text
    assert resp.text != "NICHT_GEFUNDEN"


async def test_is_deterministic() -> None:
    client = FakeLlmClient()
    kw = {"system": "x", "user": "[S1] foo", "temperature": 0.0, "max_tokens": 100}

    first = await client.complete(**kw)
    second = await client.complete(**kw)

    assert first == second


async def test_no_markers_means_nicht_gefunden() -> None:
    client = FakeLlmClient()

    resp = await client.complete(
        system="x", user="Quellen: (keine)\n\nFrage: ...", temperature=0.0, max_tokens=100
    )

    assert resp.text == "NICHT_GEFUNDEN"


async def test_repeated_markers_are_deduped_in_the_reply() -> None:
    client = FakeLlmClient()

    resp = await client.complete(
        system="x", user="[S1] a [S1] b [S2] c", temperature=0.0, max_tokens=100
    )

    assert resp.text.count("[S2]") == 1


async def test_canned_reply_overrides_everything() -> None:
    client = FakeLlmClient(canned="NICHT_GEFUNDEN")

    resp = await client.complete(
        system="x", user="[S1] [S2] [S3]", temperature=0.0, max_tokens=100
    )

    assert resp.text == "NICHT_GEFUNDEN"


async def test_markers_mentioned_only_in_the_prompt_rules_are_ignored() -> None:
    """Regression: the real answer_de.v1 prompt's own rule text says
    "...mit den Quellenmarkierungen [S1], [S2], ..." unconditionally. With
    only one real source, scanning the whole message used to make this
    "cite" a [S2] that was never offered."""
    client = FakeLlmClient()
    user = (
        "Belege jede Aussage mit den Markierungen [S1], [S2], ...\n\n"
        "Quellen:\n[S1] Quelle: a.pdf | Seite 1\ntext\n\n"
        "Frage: Was gilt?\n\n"
        "Antwort:"
    )

    resp = await client.complete(system="x", user=user, temperature=0.0, max_tokens=100)

    assert "[S1]" in resp.text
    assert "[S2]" not in resp.text


async def test_an_empty_context_block_between_quellen_and_frage_is_nicht_gefunden() -> None:
    client = FakeLlmClient()
    user = (
        "Belege jede Aussage mit den Markierungen [S1], [S2], ...\n\n"
        "Quellen:\n\n\nFrage: Was gilt?\n\nAntwort:"
    )

    resp = await client.complete(system="x", user=user, temperature=0.0, max_tokens=100)

    assert resp.text == "NICHT_GEFUNDEN"


async def test_stream_reassembles_to_the_same_text_complete_would_give() -> None:
    client = FakeLlmClient()
    kw = {"system": "x", "user": "[S1] Quelle: a.pdf\n...", "temperature": 0.0, "max_tokens": 100}

    deltas = [delta async for delta in client.stream(**kw)]
    whole = await client.complete(**kw)

    assert "".join(d.text for d in deltas) == whole.text
    assert len(deltas) > 1, "should actually be chunked, not one big piece"
    assert all(d.model == "fake" for d in deltas)
    assert all(d.prompt_tokens is None and d.completion_tokens is None for d in deltas)


async def test_stream_honours_a_canned_reply_too() -> None:
    client = FakeLlmClient(canned="NICHT_GEFUNDEN")

    deltas = [
        delta
        async for delta in client.stream(
            system="x", user="[S1] [S2]", temperature=0.0, max_tokens=100
        )
    ]

    assert "".join(d.text for d in deltas) == "NICHT_GEFUNDEN"
