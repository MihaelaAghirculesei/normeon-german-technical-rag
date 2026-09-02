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


settings = Settings()
