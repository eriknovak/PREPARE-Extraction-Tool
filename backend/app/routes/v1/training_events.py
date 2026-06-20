"""Internal callback endpoint receiving training lifecycle events from the trainer."""

import logging
from typing import List

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.core.database import get_session
from app.models_db import Record, TrainingRun
from app.services import training_service
from app.services.websocket_manager import manager

logger = logging.getLogger(__name__)
router = APIRouter()


def _safe_float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@router.post("/internal/training-events")
async def receive_training_event(payload: dict, db: Session = Depends(get_session)):
    """Handle a training lifecycle event posted by the bioner trainer.

    Recognized ``type`` values: training_info, epoch_update,
    baseline_evaluation_completed, evaluation_completed, completed, stopped, error.
    """
    event_type = payload.get("type")
    run_id = payload.get("run_id")
    if not run_id:
        return {"ok": False, "error": "missing run_id"}

    if event_type == "training_info":
        training_service.mark_running(db, run_id)
    elif event_type == "epoch_update":
        # Only persist points that carry a loss value; loss-less epoch-boundary
        # ticks are still broadcast below for the live UI.
        loss = _safe_float(payload.get("loss"))
        if loss is not None:
            epoch_raw = payload.get("epoch")
            epoch = int(float(epoch_raw)) if epoch_raw is not None else 0
            training_service.add_epoch_metric(db, run_id, epoch=epoch, loss=loss)
    elif event_type == "baseline_evaluation_completed":
        metrics = payload.get("metrics") or {}
        per_label = metrics.get("per_label") or {}
        training_service.record_baseline_evaluation(db, run_id, per_label)
    elif event_type == "evaluation_completed":
        metrics = payload.get("metrics") or {}
        per_label = metrics.get("per_label") or {}
        training_service.record_evaluation(db, run_id, per_label)
    elif event_type == "completed":
        run = db.get(TrainingRun, run_id)
        record_ids: List[int] = []
        if run is not None:
            # Link every record across the run's training datasets. Fall back to
            # the primary dataset if no link rows exist (legacy/back-compat).
            train_dataset_ids = training_service.get_dataset_ids(
                db, run_id, role="train"
            ) or [run.dataset_id]
            record_ids = list(
                db.exec(
                    select(Record.id).where(
                        Record.dataset_id.in_(train_dataset_ids)
                    )
                ).all()
            )
        training_service.complete_run(
            db, run_id, output_path=payload.get("output_path"), record_ids=record_ids
        )
    elif event_type == "stopped":
        training_service.stop_run(db, run_id)
    elif event_type == "error":
        training_service.fail_run(db, run_id, payload.get("message", "training failed"))

    try:
        await manager.broadcast(payload)
    except Exception as exc:  # broadcast is best-effort
        logger.warning("broadcast failed: %s", exc)
    return {"ok": True}
