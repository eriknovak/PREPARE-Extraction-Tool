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
        BACKEND_URL (str): Base URL of the backend service used for training
            event callbacks.
    """

    BACKEND_URL: str = "http://prepare-backend:8000"

    # ======================================================
    # Environment setting
    # ======================================================

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
