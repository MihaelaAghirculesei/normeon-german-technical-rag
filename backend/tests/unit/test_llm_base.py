from app.adapters.llm.base import is_truncated


def test_only_a_normal_stop_or_an_unreported_reason_counts_as_finished() -> None:
    assert not is_truncated(None)
    assert not is_truncated("stop")
    assert not is_truncated("STOP")
    assert is_truncated("length")
    assert is_truncated("content_filter")
    assert is_truncated("max_tokens")
