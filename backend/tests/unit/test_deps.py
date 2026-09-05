"""Unit tests for the misconfiguration paths of the get_* dependency
factories in api/deps.py -- the happy paths are exercised indirectly by
every other test that uses EmbedderDep/RerankerDep/LlmClientDep.
"""

import pytest

from app.api import deps
from app.core.errors import EmbeddingMisconfiguredError, LlmMisconfiguredError


def test_get_llm_client_raises_a_typed_error_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deps.settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(deps.settings, "llm_api_base_url", None)
    monkeypatch.setattr(deps.settings, "llm_model", "some-model")
    deps.get_llm_client.cache_clear()

    try:
        with pytest.raises(LlmMisconfiguredError) as exc_info:
            deps.get_llm_client()
        assert exc_info.value.code == "llm_misconfigured"
        assert exc_info.value.status_code == 500
    finally:
        deps.get_llm_client.cache_clear()


def test_get_embedder_raises_a_typed_error_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deps.settings, "embedding_provider", "e5_api")
    monkeypatch.setattr(deps.settings, "embedding_api_base_url", None)
    deps.get_embedder.cache_clear()

    try:
        with pytest.raises(EmbeddingMisconfiguredError) as exc_info:
            deps.get_embedder()
        assert exc_info.value.code == "embedding_misconfigured"
    finally:
        deps.get_embedder.cache_clear()
