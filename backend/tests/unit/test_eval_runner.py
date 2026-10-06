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
from app.eval.models import EvalConfig, EvalQuestion, EvalReport, QuestionRun
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


def _answer(
    text: str = "Laut [S1]. [S1]", cost: float | None = 0.0002, invented: int = 0
) -> AnswerResult:
    return AnswerResult(
        answer=text, sources=[_source()], citations=[_citation()],
        prompt_name="answer_de.v1", prompt_sha256="a" * 64, model="fake",
        retrieval_timing=PipelineTiming(10.0, 5.0, 1.0, 16.0),
        generation_ms=3.0, prompt_tokens=None, completion_tokens=None, cost_usd=cost,
        invented_citations=invented,
    )


def _stub_generate(monkeypatch: Any, by_id: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_generate_answer(
        session: Any, embedder: Any, reranker: Any, llm: Any, *,
        tenant_id: uuid.UUID, question: str, strategy: str | None = None,
        retrieval_mode: str | None = None, rerank_top_k: int | None = None,
        candidate_k: int | None = None,
        prompt_name: str | None = None, request_id: str | None = None,
    ) -> AnswerResult:
        calls.append({
            "question": question, "strategy": strategy, "request_id": request_id,
            "retrieval_mode": retrieval_mode, "rerank_top_k": rerank_top_k,
            "candidate_k": candidate_k,
        })
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
    assert report.config_hash == compute_config_hash(**CONFIG.model_dump(exclude_none=True))
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


# --- checkpoint / resume ----------------------------------------------------


async def test_checkpoint_records_every_finished_question(
    monkeypatch: Any, tmp_path: Path,
) -> None:
    _stub_generate(monkeypatch, {"a": _answer(), "b": RuntimeError("down")})
    path = tmp_path / "run.partial.jsonl"

    await runner.run_evaluation(
        [_q("Q001", question="a"), _q("Q002", question="b")], CONFIG,
        session_factory=_session_factory,  # type: ignore[arg-type]
        embedder=object(), reranker=object(), llm=object(), tenant_id=TENANT,
        checkpoint=path,
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert sorted(QuestionRun.model_validate_json(x).question_id for x in lines) == [
        "Q001", "Q002",
    ]


async def test_resume_skips_completed_questions_and_retries_errored_ones(
    monkeypatch: Any, tmp_path: Path,
) -> None:
    path = tmp_path / "run.partial.jsonl"
    questions = [_q("Q001", question="a"), _q("Q002", question="b"), _q("Q003", question="c")]
    _stub_generate(monkeypatch, {
        "a": _answer("A [S1]"), "b": RuntimeError("down"), "c": _answer("C [S1]"),
    })
    await runner.run_evaluation(
        questions, CONFIG, session_factory=_session_factory,  # type: ignore[arg-type]
        embedder=object(), reranker=object(), llm=object(), tenant_id=TENANT,
        checkpoint=path,
    )

    calls = _stub_generate(monkeypatch, {"b": _answer("B [S1]")})
    report = await runner.run_evaluation(
        questions, CONFIG, session_factory=_session_factory,  # type: ignore[arg-type]
        embedder=object(), reranker=object(), llm=object(), tenant_id=TENANT,
        checkpoint=path,
    )

    assert [c["question"] for c in calls] == ["b"]
    assert [r.question_id for r in report.results] == ["Q001", "Q002", "Q003"]
    assert [r.answer for r in report.results] == ["A [S1]", "B [S1]", "C [S1]"]
    assert all(r.error is None for r in report.results)


async def test_resume_tolerates_a_torn_last_line_and_ignores_unknown_ids(
    monkeypatch: Any, tmp_path: Path,
) -> None:
    path = tmp_path / "run.partial.jsonl"
    _stub_generate(monkeypatch, {"a": _answer("A [S1]")})
    await runner.run_evaluation(
        [_q("Q001", question="a")], CONFIG,
        session_factory=_session_factory,  # type: ignore[arg-type]
        embedder=object(), reranker=object(), llm=object(), tenant_id=TENANT,
        checkpoint=path,
    )
    stale = path.read_text(encoding="utf-8").replace('"Q001"', '"Q999"')
    with path.open("a", encoding="utf-8") as fh:
        fh.write(stale)
        fh.write('{"question_id": "Q002", "categ')  # died mid-write

    calls = _stub_generate(monkeypatch, {"b": _answer("B [S1]")})
    report = await runner.run_evaluation(
        [_q("Q001", question="a"), _q("Q002", question="b")], CONFIG,
        session_factory=_session_factory,  # type: ignore[arg-type]
        embedder=object(), reranker=object(), llm=object(), tenant_id=TENANT,
        checkpoint=path,
    )

    assert [c["question"] for c in calls] == ["b"]
    assert [r.answer for r in report.results] == ["A [S1]", "B [S1]"]


async def test_rerank_candidates_is_forwarded_as_candidate_k(monkeypatch: Any) -> None:
    calls = _stub_generate(monkeypatch, {"x": _answer()})
    config = CONFIG.model_copy(update={"rerank_candidates": 10})

    await runner.run_evaluation(
        [_q("Q001", question="x")], config,
        session_factory=_session_factory,  # type: ignore[arg-type]
        embedder=object(), reranker=object(), llm=object(), tenant_id=TENANT,
    )

    assert calls[0]["candidate_k"] == 10


def test_unset_new_knobs_leave_the_config_hash_unchanged() -> None:
    """Reports written before rerank_candidates/reranker_max_length
    existed keep the hash they were filed under."""
    new_knobs = {"rerank_candidates", "reranker_max_length"}
    legacy = compute_config_hash(**CONFIG.model_dump(exclude=new_knobs))
    assert runner.config_hash(CONFIG) == legacy
    assert runner.config_hash(CONFIG.model_copy(update={"rerank_candidates": 10})) != legacy


def test_checkpoint_path_separates_question_sets_that_share_ids(tmp_path: Path) -> None:
    smoke = [_q("Q001", question="Smoke-Frage?")]
    real = [_q("Q001", question="Echte Frage?")]

    assert runner.checkpoint_path(CONFIG, smoke, tmp_path) != runner.checkpoint_path(
        CONFIG, real, tmp_path
    )
    assert runner.checkpoint_path(CONFIG, real, tmp_path) == runner.checkpoint_path(
        CONFIG, list(real), tmp_path
    )
    assert runner.checkpoint_path(CONFIG, real, tmp_path).name.startswith(
        runner.config_hash(CONFIG)
    )
