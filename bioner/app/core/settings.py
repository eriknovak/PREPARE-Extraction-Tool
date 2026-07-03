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
        TRAINING_STOP_JOIN_TIMEOUT (float): Seconds to wait for a stop-requested
            training worker to wind down before a new run is reported as still
            stopping (409 TRAINING_STOPPING). Kept short so the API stays
            responsive; the client retries.
    """

    BACKEND_HOST: str = "http://localhost:8000"

    TRAINING_STOP_JOIN_TIMEOUT: float = 5.0

    # ======================================================
    # Environment setting
    # ======================================================

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
