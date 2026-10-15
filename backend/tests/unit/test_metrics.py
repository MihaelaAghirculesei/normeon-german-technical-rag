"""Prometheus metrics (Giorno 27): the `/metrics` endpoint and what
generation records in it."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.adapters.llm.base import LlmResponse
from app.core.config import settings
from app.core.errors import LlmUnavailableError
from app.core.metrics import sample
from app.main import app
from app.services import generation
from tests.unit.test_generation import TENANT, _chunk, _stub_retrieval

# -- /metrics -------------------------------------------------------------------


def test_metrics_are_exposed_for_prometheus() -> None:
    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    for series in (
        "normeon_phase_seconds",
        "normeon_answers_total",
        "normeon_llm_cost_usd_total",
        "normeon_hallucinated_citations_total",
        "normeon_llm_errors_total",
    ):
        assert f"# HELP {series}" in response.text


def test_a_metrics_token_closes_the_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "metrics_token", "scrape-me")
    client = TestClient(app)

    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer scrape-me"}).status_code == 200


# -- what generation records ----------------------------------------------------


class _Llm:
    name = "spy"

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error

    async def complete(
        self, *, system: str, user: str, temperature: float, max_tokens: int
    ) -> LlmResponse:
        if self.error is not None:
            raise self.error
        return LlmResponse(text=self.text, model="spy")


async def _answer(llm: _Llm) -> Any:
    return await generation.generate_answer(
        object(), object(), object(), llm, tenant_id=TENANT, question="Frage?"
    )


async def test_an_answer_is_counted_with_its_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1")])
    answered = sample("normeon_answers_total", {"outcome": "answered"})
    citations = sample("normeon_citations_total")

    await _answer(_Llm("Es gilt Regel A [S1]."))

    assert sample("normeon_answers_total", {"outcome": "answered"}) == answered + 1
    assert sample("normeon_citations_total") == citations + 1


async def test_an_abstention_is_counted_as_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1")])
    abstained = sample("normeon_answers_total", {"outcome": "abstained"})

    await _answer(_Llm("NICHT_GEFUNDEN"))

    assert sample("normeon_answers_total", {"outcome": "abstained"}) == abstained + 1


async def test_an_llm_failure_is_counted_per_provider_and_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_retrieval(monkeypatch, [_chunk("5.1")])
    labels = {"provider": settings.llm_provider, "code": "llm_unavailable"}
    before = sample("normeon_llm_errors_total", labels)

    with pytest.raises(LlmUnavailableError):
        await _answer(_Llm(error=LlmUnavailableError("host down")))

    assert sample("normeon_llm_errors_total", labels) == before + 1
