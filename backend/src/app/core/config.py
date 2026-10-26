from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def asyncpg_url(url: str) -> str:
    """The URL in the form SQLAlchemy's asyncpg dialect accepts. Hosted
    Postgres hands out libpq URLs: a `postgres://` or `postgresql://`
    scheme, `sslmode=require` (asyncpg's keyword is `ssl`) and
    `channel_binding`, which asyncpg rejects as an unknown argument."""
    parts = urlsplit(url)
    scheme = parts.scheme
    if scheme in {"postgres", "postgresql"}:
        scheme = "postgresql+asyncpg"
    if scheme != "postgresql+asyncpg":
        return url
    query: list[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "sslmode":
            query.append(("ssl", value))
        elif key != "channel_binding":
            query.append((key, value))
    return urlunsplit(parts._replace(scheme=scheme, query=urlencode(query)))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "normeon"
    environment: str = "development"
    # A provider's URL can be pasted as given (e.g. Neon's
    # `postgresql://...?sslmode=require`): see `asyncpg_url`.
    database_url: str = "postgresql+asyncpg://normeon:normeon@localhost:5432/normeon"
    log_level: str = "INFO"
    # The built frontend (`dist/frontend/browser`). When set, the API also
    # serves the app, so a deployment is one origin (core/spa.py).
    static_dir: Path | None = None

    @field_validator("database_url")
    @classmethod
    def _asyncpg_database_url(cls, value: str) -> str:
        return asyncpg_url(value)

    # `gemini` embeds at Google (set `embedding_model=gemini-embedding-2`
    # and `embedding_api_key`), so the process needs no torch (ADR 0005).
    embedding_provider: Literal["e5_local", "e5_api", "gemini"] = "e5_local"
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: int = 1024
    embedding_api_base_url: str | None = None
    embedding_api_key: str | None = None

    # Retrieval. `retrieval_strategy` picks which chunking to search over
    # (the two are ingested side by side); the eval matrix in Week 4 flips
    # it. `hnsw_ef_search` is pgvector's per-session recall/latency knob --
    # below ~20 recall collapses, above ~100 costs latency for nothing.
    retrieval_strategy: Literal["fixed_500", "structural"] = "structural"
    retrieval_top_k: int = 10
    hnsw_ef_search: int = 40
    # Cutoff for the trigram fallback branch that runs when a query names a
    # code ("LH-3.2.1", "UN R79"): pg_trgm.word_similarity_threshold for
    # `:code <% content_norm`. 0.5 tolerates a little spacing/spelling drift.
    trgm_code_threshold: float = 0.5
    # Full-text match on all of the question's terms or on any of them
    # (Giorno 28). "all" finds nothing for a question that names one word
    # the text does not use; "any" lets ts_rank_cd sort it out.
    fts_match: Literal["all", "any"] = "all"

    # Hybrid retrieval (Day 8): fuse the vector, full-text and trigram
    # rankings with Reciprocal Rank Fusion. `hybrid_candidate_k` is how
    # many hits to pull from each branch before fusing; `hybrid_top_k` how
    # many fused hits to keep (the Day 9 reranker narrows further, so this
    # stays wide). `rrf_k` dampens the weight of top ranks -- 60 is the
    # RRF paper's value. The three weights scale each branch's
    # contribution; all four are experiment-matrix variables in Week 4.
    hybrid_candidate_k: int = 40
    hybrid_top_k: int = 40
    rrf_k: int = 60
    rrf_weight_vector: float = 1.0
    rrf_weight_fts: float = 1.0
    rrf_weight_trgm: float = 1.0

    # Reranking + context selection (Day 9). The reranker re-scores the
    # hybrid candidates jointly against the query; `reranker_provider =
    # "noop"` is the matrix's "off" cell. `rerank_top_k` is how many
    # survive the rerank. Context selection then fills a token budget
    # (not a fixed chunk count), skipping a `section_path` already
    # represented. All four are experiment-matrix variables in Week 4.
    reranker_provider: Literal["noop", "cross_encoder"] = "cross_encoder"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Token cap for one (query, chunk) pair. Without it sentence-
    # transformers falls back to the tokenizer's own limit -- 8192 for
    # bge-reranker-v2-m3 -- so every chunk is scored in full and cost
    # grows with chunk length. 512 is the model authors' own default
    # (FlagEmbedding's `compute_score`) and what docs/FAILURE-MODES.md
    # describes; a matrix variable like the rest of this block.
    reranker_max_length: int = 512
    rerank_top_k: int = 8
    context_token_budget: int = 4000

    # Generation (Day 11). The answer prompt is a versioned file in
    # backend/prompts/, loaded by `answer_prompt_name`; its sha256 is
    # logged with every answer. The LLM seam is deliberately thin -- one
    # OpenAI-compatible wire format reaches most hosted models -- with the
    # multi-provider Protocol and cost tracking deferred to Day 15.
    # `llm_provider = "fake"` is the offline default: a deterministic
    # canned answer, no network, no key. `openai_compatible` needs
    # `llm_api_base_url` (+ usually `llm_api_key`) and `llm_model`.
    llm_provider: Literal["fake", "openai_compatible"] = "fake"
    llm_model: str = ""
    llm_api_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    llm_timeout_s: float = 30.0
    llm_call_deadline_s: float = 120.0
    llm_stream_first_token_s: float = 60.0
    llm_stream_idle_s: float = 30.0
    llm_max_retries: int = 2
    answer_prompt_name: str = "answer_de.v2"

    # Abstention + resilience (Day 13). If the top reranked chunk's score
    # is below this, `generate_answer` answers NICHT_GEFUNDEN without
    # calling the LLM at all -- the cost-saving "abstained_pre_generation"
    # path. NOTE: the raw score's scale depends on `reranker_provider` --
    # a cross-encoder logit and an RRF-fused score are not comparable, so
    # this threshold only means something once tuned per reranker (see
    # docs/FAILURE-MODES.md); 0.0 is a placeholder that only ever screens
    # out a genuinely negative cross-encoder score or an empty result,
    # same "needs eval-driven tuning" status as `rrf_k` and the rerank
    # weights. `llm_max_retries` bounds the OpenAI-compatible adapter's
    # `tenacity` retry budget for transient failures (timeouts, connection
    # errors, 5xx) -- a 4xx is never retried. `llm_timeout_s` bounds each
    # network read, `llm_call_deadline_s` a whole attempt; a stream has no
    # total cap, only `llm_stream_first_token_s` to its first delta and
    # `llm_stream_idle_s` between deltas.
    min_rerank_score_for_answer: float = 0.0

    # Auth (Giorno 25). The default secret only exists so a fresh clone runs;
    # outside `environment = "development"` / "test" startup refuses it (see
    # `insecure_settings`). Tokens are short-lived on purpose: there is no
    # refresh flow, so a working day is the trade-off between convenience
    # and how long a leaked token stays usable.
    jwt_secret: str = "dev-only-insecure-secret-change-me"
    jwt_issuer: str = "normeon"
    jwt_ttl_minutes: int = 8 * 60
    # Password for the two demo users `python -m app.db.seed` creates.
    demo_user_password: str = "normeon-demo"

    # Upload limits (Giorno 26). A body over `max_upload_mb` is refused
    # before it is read; a file is opened and checked (encryption, page
    # count, zip-bomb shape) before anything is stored; parsing in the
    # background gives up after `parse_timeout_s`.
    max_upload_mb: int = 50
    max_request_kb: int = 256  # every other endpoint takes small JSON
    max_document_pages: int = 2000
    max_docx_uncompressed_mb: int = 200
    max_docx_compression_ratio: int = 100
    upload_check_timeout_s: float = 20.0
    parse_timeout_s: float = 300.0
    # OCR for scanned pages: a PDF page whose text layer has fewer than
    # `ocr_min_chars` characters is read with tesseract instead.
    ocr_enabled: bool = True
    ocr_language: str = "deu"
    ocr_dpi: int = 300
    ocr_min_chars: int = 50

    # Observability (Giorno 27). Traces: `console` logs one readable line per
    # request; `otlp` sends them to a collector at `otel_endpoint`.
    # `/metrics` is open when `metrics_token` is unset (a scraper on the
    # internal network); set it to require `Authorization: Bearer <token>`.
    otel_exporter: Literal["console", "otlp", "none"] = "console"
    otel_endpoint: str = "http://localhost:4318/v1/traces"
    metrics_token: str | None = None


settings = Settings()

_DEV_ENVIRONMENTS = frozenset({"development", "test"})


def insecure_settings(s: Settings) -> list[str]:
    """Settings that are fine on a laptop and dangerous anywhere else."""
    if s.environment in _DEV_ENVIRONMENTS:
        return []
    problems = []
    if s.jwt_secret == Settings.model_fields["jwt_secret"].default or len(s.jwt_secret) < 32:
        problems.append("JWT_SECRET must be set to a random value of at least 32 characters")
    return problems
