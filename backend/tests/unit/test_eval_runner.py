"""Unit tests for app/eval/runner. `generate_answer` is stubbed on the
runner module, so no DB, retrieval or model is touched.
"""

import uuid
from pathlib import Path
from typing import Any

from app.domain.citations import Citation
from app.domain.config_hash import compute_config_hash
from app.domain.context import Source
from app.domain.models import PipelineTiming
from app.eval import runner
from app.eval.models import EvalConfig, EvalQuestion, EvalReport
from app.services.generation import AnswerResult

TENANT = uuid.uuid4()

CONFIG = EvalConfig(
    chunking_strategy="structural",
    reranker="noop",
    top_k=5,
    embedding_provider="e5_local",
    llm_provider="fake",
    llm_model="",
)


class _FakeSession:
    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _session_factory() -> _FakeSession:
    return _FakeSession()


def _q(qid: str, category: str = "requirement_lookup", question: str = "Frage?") -> EvalQuestion:
    return EvalQuestion.model_validate({"id": qid, "category": category, "question": question})


def _source() -> Source:
    return Source(
        marker="S1", chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), filename="StVZO.pdf",
        page_from=14, page_to=14, section_path="50", heading=None, content="…",
    )


def _citation() -> Citation:
    return Citation(
        marker="S1", chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), filename="StVZO.pdf",
        page_from=14, page_to=14, section_path="50", snippet="…",
    )


def _answer(text: str = "Laut [S1]. [S1]", cost: float | None = 0.0002) -> AnswerResult:
    return AnswerResult(
        answer=text, sources=[_source()], citations=[_citation()],
        prompt_name="answer_de.v1", prompt_sha256="a" * 64, model="fake",
        retrieval_timing=PipelineTiming(10.0, 5.0, 1.0, 16.0),
        generation_ms=3.0, prompt_tokens=None, completion_tokens=None, cost_usd=cost,
    )


def _stub_generate(monkeypatch: Any, by_id: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_generate_answer(
        session: Any, embedder: Any, reranker: Any, llm: Any, *,
        tenant_id: uuid.UUID, question: str, strategy: str | None = None,
        prompt_name: str | None = None, request_id: str | None = None,
    ) -> AnswerResult:
        calls.append({"question": question, "strategy": strategy, "request_id": request_id})
        outcome = by_id[question]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(runner, "generate_answer", fake_generate_answer)
    return calls


async def test_run_evaluation_builds_a_report_in_question_order(monkeypatch: Any) -> None:
    questions = [
        _q("Q001", question="a"),
        _q("Q002", "unanswerable", "b"),
        _q("Q003", question="c"),
    ]
    _stub_generate(monkeypatch, {
        "a": _answer("Antwort A [S1]"),
        "b": _answer("NICHT_GEFUNDEN"),
        "c": _answer("Antwort C [S1]"),
    })

    report = await runner.run_evaluation(
        questions, CONFIG, session_factory=_session_factory,  # type: ignore[arg-type]
        embedder=object(), reranker=object(), llm=object(), tenant_id=TENANT,
    )

    assert isinstance(report, EvalReport)
    assert report.n_questions == 3
    assert report.config_hash == compute_config_hash(**CONFIG.model_dump())
    assert [r.question_id for r in report.results] == ["Q001", "Q002", "Q003"]
    assert [r.abstained for r in report.results] == [False, True, False]
    assert report.results[0].category == "requirement_lookup"
    assert report.results[0].cost_usd == 0.0002
    assert report.results[0].total_ms == 19.0
    assert [rr.filename for rr in report.results[0].retrieved] == ["StVZO.pdf"]


async def test_a_failing_question_is_captured_not_fatal(monkeypatch: Any) -> None:
    questions = [_q("Q001", question="ok"), _q("Q002", question="boom")]
    _stub_generate(monkeypatch, {"ok": _answer(), "boom": RuntimeError("provider exploded")})

    report = await runner.run_evaluation(
        questions, CONFIG, session_factory=_session_factory,  # type: ignore[arg-type]
        embedder=object(), reranker=object(), llm=object(), tenant_id=TENANT,
    )

    assert report.results[0].error is None
    assert report.results[1].error == "provider exploded"
    assert report.results[1].answer == ""
    assert report.results[1].abstained is False


async def test_request_id_carries_the_run_hash_and_question_id(monkeypatch: Any) -> None:
    calls = _stub_generate(monkeypatch, {"x": _answer()})

    report = await runner.run_evaluation(
        [_q("Q042", question="x")], CONFIG, session_factory=_session_factory,  # type: ignore[arg-type]
        embedder=object(), reranker=object(), llm=object(), tenant_id=TENANT,
    )

    assert calls[0]["strategy"] == "structural"
    assert calls[0]["request_id"] == f"eval-{report.config_hash[:12]}-Q042"


async def test_write_report_names_the_file_by_config_hash_and_round_trips(
    monkeypatch: Any, tmp_path: Path,
) -> None:
    _stub_generate(monkeypatch, {"x": _answer()})
    report = await runner.run_evaluation(
        [_q("Q001", question="x")], CONFIG, session_factory=_session_factory,  # type: ignore[arg-type]
        embedder=object(), reranker=object(), llm=object(), tenant_id=TENANT,
    )

    path = runner.write_report(report, tmp_path)

    assert path.name == f"{report.config_hash}.json"
    assert EvalReport.model_validate_json(path.read_text(encoding="utf-8")) == report
