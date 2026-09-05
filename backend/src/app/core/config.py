from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "normeon"
    environment: str = "development"
    database_url: str = "postgresql+asyncpg://normeon:normeon@localhost:5432/normeon"
    log_level: str = "INFO"

    embedding_provider: Literal["e5_local", "e5_api"] = "e5_local"
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
    llm_max_retries: int = 2
    answer_prompt_name: str = "answer_de.v1"

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
    # errors, 5xx) -- a 4xx is never retried.
    min_rerank_score_for_answer: float = 0.0


settings = Settings()
