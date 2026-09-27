from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central app config, loaded from environment / .env (C6: no secrets in git)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_env: str = "development"
    log_level: str = "INFO"
    cors_allow_origins: list[str] = ["http://localhost:3000"]

    database_url: str = "postgresql+psycopg://bugflow:bugflow@localhost:5432/bugflow"
    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "http://localhost:5000"

    jwt_secret_key: str = "change-me-in-real-env"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7

    github_app_id: str = ""
    github_app_private_key_path: str = ""
    github_webhook_secret: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
