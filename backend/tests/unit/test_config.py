from app.core.config import Settings


def test_settings_have_sane_defaults() -> None:
    settings = Settings()

    assert settings.app_name == "normeon"
    assert settings.database_url.startswith("postgresql+asyncpg://")
