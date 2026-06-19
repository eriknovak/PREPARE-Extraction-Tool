"""Training run lifecycle: create, progress, evaluate, complete/fail/stop."""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlmodel import Session, select

from app.models_db import (
    Model,
    ModelTrainRecordLink,
    TrainingMetric,
    TrainingRun,
)
from app.services import evaluation_service


def create_run(
    db: Session,
    *,
    dataset_id: int,
    base_model: str,
    labels: List[str],
    val_ratio: float,
) -> TrainingRun:
    """Create a TrainingRun in the 'pending' state.

    Args:
        db (Session): Active DB session.
        dataset_id (int): Dataset to train on.
        base_model (str): HuggingFace model identifier used as starting weights.
        labels (List[str]): NER entity labels the run will train for.
        val_ratio (float): Fraction of data held out for validation.

    Returns:
        TrainingRun: The newly created run.
    """
    run = TrainingRun(
        dataset_id=dataset_id,
        base_model=base_model,
        labels=labels,
        val_ratio=val_ratio,
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def mark_running(db: Session, run_id: int) -> None:
    """Transition a run to 'running'.

    Args:
        db (Session): Active DB session.
        run_id (int): ID of the TrainingRun to update.
    """
    run = db.get(TrainingRun, run_id)
    if run is None:
        return
    run.status = "running"
    db.add(run)
    db.commit()


def add_epoch_metric(db: Session, run_id: int, epoch: int, loss: Optional[float]) -> None:
    """Append a per-epoch loss point.

    Args:
        db (Session): Active DB session.
        run_id (int): ID of the TrainingRun this metric belongs to.
        epoch (int): Epoch number (1-indexed).
        loss (Optional[float]): Training loss for this epoch.
    """
    db.add(TrainingMetric(run_id=run_id, epoch=epoch, loss=loss))
    db.commit()


def get_run_metrics(db: Session, run_id: int) -> List[TrainingMetric]:
    """Return a run's per-epoch loss points, ordered by epoch.

    Args:
        db (Session): Active DB session.
        run_id (int): ID of the TrainingRun whose metrics to read.

    Returns:
        List[TrainingMetric]: Metric points ordered ascending by epoch.
    """
    return db.exec(
        select(TrainingMetric)
        .where(TrainingMetric.run_id == run_id)
        .order_by(TrainingMetric.epoch)
    ).all()


def _ensure_model(db: Session, run: TrainingRun) -> Model:
    """Return the run's Model, creating and linking one if absent.

    Args:
        db (Session): Active DB session.
        run (TrainingRun): The run whose model to retrieve or create.

    Returns:
        Model: The existing or newly created Model row.
    """
    if run.model_id is not None:
        model = db.get(Model, run.model_id)
        if model is not None:
            return model
    model = Model(
        name=f"run-{run.id}",
        version=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        base_model=run.base_model,
        dataset_id=run.dataset_id,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    run.model_id = model.id
    db.add(run)
    db.commit()
    return model


def record_evaluation(
    db: Session, run_id: int, per_label: Dict[str, Dict[str, float]]
) -> None:
    """Store final per-label evaluation against the run's Model.

    Args:
        db (Session): Active DB session.
        run_id (int): ID of the TrainingRun whose model to evaluate.
        per_label (Dict[str, Dict[str, float]]): Label -> metric mapping
            (e.g. {"Drug": {"exact_f1": 0.8, "relaxed_f1": 0.9}}).
    """
    run = db.get(TrainingRun, run_id)
    if run is None:
        return
    model = _ensure_model(db, run)
    evaluation_service.store_evaluation(
        db, model_id=model.id, dataset_id=run.dataset_id, per_label=per_label
    )


def complete_run(
    db: Session, run_id: int, output_path: str, record_ids: List[int]
) -> Optional[Model]:
    """Finalize a successful run: set artifact path, link training records, mark completed.

    Args:
        db (Session): Active DB session.
        run_id (int): ID of the TrainingRun to complete.
        output_path (str): Filesystem path to the saved model artifact.
        record_ids (List[int]): Record IDs used during training.

    Returns:
        Optional[Model]: The finalized Model, or None if the run was not found.
    """
    run = db.get(TrainingRun, run_id)
    if run is None:
        return None
    model = _ensure_model(db, run)
    model.path = output_path
    db.add(model)
    for record_id in record_ids:
        db.add(ModelTrainRecordLink(model_id=model.id, record_id=record_id))
    run.status = "completed"
    db.add(run)
    db.commit()
    db.refresh(model)
    return model


def fail_run(db: Session, run_id: int, message: str) -> None:
    """Mark a run failed with an error message.

    Args:
        db (Session): Active DB session.
        run_id (int): ID of the TrainingRun to fail.
        message (str): Human-readable error description.
    """
    run = db.get(TrainingRun, run_id)
    if run is None:
        return
    run.status = "failed"
    run.error_message = message
    db.add(run)
    db.commit()


def stop_run(db: Session, run_id: int) -> None:
    """Mark a run stopped.

    Args:
        db (Session): Active DB session.
        run_id (int): ID of the TrainingRun to stop.
    """
    run = db.get(TrainingRun, run_id)
    if run is None:
        return
    run.status = "stopped"
    db.add(run)
    db.commit()
