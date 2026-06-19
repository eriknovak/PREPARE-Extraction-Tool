"""Active-model state shared between the API process and the inference worker.

LitServe runs ``LitAPI.setup``/``predict`` in a separate (spawned) worker
process, while HTTP route handlers (``/model/activate``, ``/model/info``) run in
the API process. They therefore cannot share the loaded model in memory. This
module persists the *desired* active model to a small JSON state file: the
activate route writes it, and the worker reads it before each inference and
hot-swaps the in-memory engine when it changes (see ``NERAPI.predict``).
"""

import json
import logging
import os
import tempfile
from pathlib import Path

from huggingface_hub import model_info

logger = logging.getLogger(__name__)

# Shared state file. Defaults to a temp path so it resets on a fresh container;
# override with BIONER_ACTIVE_MODEL_STATE when the API and worker need a
# specific shared location.
STATE_PATH = Path(
    os.environ.get("BIONER_ACTIVE_MODEL_STATE")
    or (Path(tempfile.gettempdir()) / "bioner_active_model.json")
)


def read_model_metadata(model_dir: str) -> dict:
    """Return a model's ``metadata.json`` (name/version), or sensible defaults.

    A local model directory may ship a ``metadata.json``; HuggingFace ids and
    directories without one fall back to a generic name/version.
    """
    metadata_path = Path(model_dir) / "metadata.json"
    if not metadata_path.exists():
        return {"name": "Extraction model", "version": "1.0"}
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if "name" not in metadata or "version" not in metadata:
        raise ValueError("metadata.json must contain at least 'name' and 'version'")
    return metadata


def is_local_path(model: str) -> bool:
    """True if ``model`` points at an existing local file/directory."""
    try:
        return Path(model).exists()
    except OSError:
        return False


def validate_model(model: str) -> bool:
    """Best-effort check that a model can be loaded.

    Local paths must exist; otherwise the id is validated against the
    HuggingFace hub. Network validation is skipped for local paths so offline
    activation of locally trained models keeps working.
    """
    if is_local_path(model):
        return True
    try:
        model_info(model)
        return True
    except Exception:
        return False


def write_desired(model: str, metadata: dict) -> None:
    """Atomically record the desired active model for the worker to pick up."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_name(STATE_PATH.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"model": model, "metadata": metadata}, f)
    os.replace(tmp, STATE_PATH)


def read_desired() -> dict | None:
    """Return the desired-model state written by the activate route, if any."""
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
