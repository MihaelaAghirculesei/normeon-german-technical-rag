from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "normeon"
    environment: str = "development"
    database_url: str = "postgresql+asyncpg://normeon:normeon@localhost:5432/normeon"
    log_level: str = "INFO"


settings = Settings()
