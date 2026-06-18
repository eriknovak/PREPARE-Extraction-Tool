from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
# ======================================================
# Settings configuration
# ======================================================


class Settings(BaseSettings):
    """Bioner service settings.

    Attributes:
        BACKEND_HOST (str): Base URL of the backend service used for training
            event callbacks. Defaults to localhost for local runs; overridden to
            the compose service name (http://backend:8000) in docker-compose.
    """

    BACKEND_HOST: str = "http://localhost:8000"

    # ======================================================
    # Environment setting
    # ======================================================

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
