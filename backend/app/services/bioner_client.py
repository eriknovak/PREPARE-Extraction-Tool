from typing import Optional

import requests
from app.core.settings import settings


def get_training_status(run_id: int) -> Optional[dict]:
    """Return bioner's status snapshot for a run, or None if bioner doesn't know it.

    bioner's job manager is in-memory (and single-job), so a run it has never
    seen — or has forgotten after a restart — yields a 404, returned here as
    ``None`` (a definitive "not running"). A live run reports
    ``{"status": "running", ...}``.

    Raises ``requests.RequestException`` if bioner is unreachable — that is
    *unknown*, not "not running", so callers must not treat it as stale.
    """
    resp = requests.get(
        f"{settings.EXTRACT_HOST}/training/status/{run_id}",
        timeout=10,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def start_training(payload: dict):
    response = requests.post(
        f"{settings.EXTRACT_HOST}/training/start",
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def stop_training(run_id: int):
    response = requests.post(
        f"{settings.EXTRACT_HOST}/training/stop/{run_id}",
        timeout=10,
    )

    response.raise_for_status()

    return response.json()
