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
        BIONER_TRAIN_MAX_TOKENS (int): Hard cap on tokens per training window.
            Long records are trimmed to span-centred windows of at most this many
            tokens before reaching the GLiNER collator, bounding the
            ``seq_len x span_width x num_classes`` score tensor (which otherwise
            makes the first CPU step crawl).
    """

    BACKEND_HOST: str = "http://localhost:8000"
    BIONER_TRAIN_MAX_TOKENS: int = 256

    # ======================================================
    # Environment setting
    # ======================================================

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
