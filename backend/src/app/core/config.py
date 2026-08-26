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


settings = Settings()
