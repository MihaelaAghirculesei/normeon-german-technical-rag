from app.core.config import Settings, asyncpg_url


def test_settings_have_sane_defaults() -> None:
    settings = Settings()

    assert settings.app_name == "normeon"
    assert settings.database_url.startswith("postgresql+asyncpg://")


def test_a_hosted_libpq_url_is_turned_into_an_asyncpg_url() -> None:
    neon = (
        "postgresql://owner:pw@ep-x.eu-central-1.aws.neon.tech/neondb"
        "?sslmode=require&channel_binding=require"
    )

    assert asyncpg_url(neon) == (
        "postgresql+asyncpg://owner:pw@ep-x.eu-central-1.aws.neon.tech/neondb?ssl=require"
    )
    assert asyncpg_url("postgres://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"


def test_an_asyncpg_url_is_left_alone() -> None:
    url = "postgresql+asyncpg://normeon:normeon@localhost:5433/normeon"

    assert asyncpg_url(url) == url
    assert Settings(database_url=url).database_url == url


def test_settings_normalise_the_database_url_they_are_given() -> None:
    settings = Settings(database_url="postgres://u:p@h/db?sslmode=require")

    assert settings.database_url == "postgresql+asyncpg://u:p@h/db?ssl=require"
