"""Training run lifecycle: create, progress, evaluate, complete/fail/stop."""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlmodel import Session, select

from app.models_db import (
    Model,
    ModelTrainRecordLink,
    TrainingMetric,
    TrainingRun,
    TrainingRunDatasetLink,
)
from app.services import evaluation_service


def create_run(
    db: Session,
    *,
    dataset_ids: List[int],
    base_model: str,
    labels: List[str],
    val_ratio: float,
    eval_dataset_ids: Optional[List[int]] = None,
) -> TrainingRun:
    """Create a TrainingRun in the 'pending' state.

    The first id in ``dataset_ids`` becomes the run's primary training dataset
    (``TrainingRun.dataset_id``); all training and eval datasets are recorded in
    ``training_run_dataset_link``.

    Args:
        db (Session): Active DB session.
        dataset_ids (List[int]): Datasets to train on (first = primary).
        base_model (str): HuggingFace model identifier used as starting weights.
        labels (List[str]): NER entity labels the run will train for.
        val_ratio (float): Fraction of data held out for validation.
        eval_dataset_ids (Optional[List[int]]): Datasets to evaluate against
            instead of a held-out split (optional).

    Returns:
        TrainingRun: The newly created run.
    """
    run = TrainingRun(
        dataset_id=dataset_ids[0],
        base_model=base_model,
        labels=labels,
        val_ratio=val_ratio,
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    for dsid in dataset_ids:
        db.add(
            TrainingRunDatasetLink(
                training_run_id=run.id, dataset_id=dsid, role="train"
            )
        )
    for dsid in eval_dataset_ids or []:
        db.add(
            TrainingRunDatasetLink(
                training_run_id=run.id, dataset_id=dsid, role="eval"
            )
        )
    db.commit()
    return run


def get_dataset_ids(db: Session, run_id: int, role: str = "train") -> List[int]:
    """Return the dataset ids linked to a run for the given role.

    Args:
        db (Session): Active DB session.
        run_id (int): The training run id.
        role (str): ``"train"`` or ``"eval"``.

    Returns:
        List[int]: Linked dataset ids (empty if none recorded).
    """
    return list(
        db.exec(
            select(TrainingRunDatasetLink.dataset_id)
            .where(TrainingRunDatasetLink.training_run_id == run_id)
            .where(TrainingRunDatasetLink.role == role)
        ).all()
    )


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


def has_active_run(db: Session, dataset_id: int) -> bool:
    """Return True if the dataset has a pending or running training run.

    Args:
        db (Session): Active DB session.
        dataset_id (int): Dataset to check.

    Returns:
        bool: Whether an active (pending/running) run exists.
    """
    active = db.exec(
        select(TrainingRun)
        .where(TrainingRun.dataset_id == dataset_id)
        .where(TrainingRun.status.in_(["pending", "running"]))
    ).first()
    return active is not None


def has_active_run_for_datasets(db: Session, dataset_ids: List[int]) -> bool:
    """Return True if any of the given datasets has an active training run.

    Checks the training-dataset links so multi-dataset runs are matched on any
    of their training datasets, not just the primary one.

    Args:
        db (Session): Active DB session.
        dataset_ids (List[int]): Datasets to check.

    Returns:
        bool: Whether an active (pending/running) run uses any of the datasets.
    """
    if not dataset_ids:
        return False
    active = db.exec(
        select(TrainingRun.id)
        .join(
            TrainingRunDatasetLink,
            TrainingRunDatasetLink.training_run_id == TrainingRun.id,
        )
        .where(TrainingRunDatasetLink.role == "train")
        .where(TrainingRunDatasetLink.dataset_id.in_(dataset_ids))
        .where(TrainingRun.status.in_(["pending", "running"]))
    ).first()
    return active is not None


def update_run(
    db: Session,
    run_id: int,
    *,
    name: Optional[str] = None,
    preferred: Optional[bool] = None,
) -> Optional[TrainingRun]:
    """Rename a run and/or designate it as the dataset's preferred run.

    Setting ``preferred=True`` clears the flag on every other run in the same
    dataset so at most one run is preferred per dataset.

    Args:
        db (Session): Active DB session.
        run_id (int): ID of the TrainingRun to update.
        name (Optional[str]): New display name (empty string clears it).
        preferred (Optional[bool]): New preferred flag.

    Returns:
        Optional[TrainingRun]: The updated run, or None if not found.
    """
    run = db.get(TrainingRun, run_id)
    if run is None:
        return None
    if name is not None:
        stripped = name.strip()
        run.name = stripped or None
    if preferred is not None:
        if preferred:
            others = db.exec(
                select(TrainingRun)
                .where(TrainingRun.dataset_id == run.dataset_id)
                .where(TrainingRun.id != run.id)
                .where(TrainingRun.preferred)  # noqa: E712
            ).all()
            for other in others:
                other.preferred = False
                db.add(other)
        run.preferred = preferred
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def delete_run(db: Session, run_id: int) -> bool:
    """Delete a run and its dependent rows.

    Cascade behaviour: the run's TrainingMetric rows are removed via the
    relationship cascade. If the run produced a Model, that Model and its
    dependents (Evaluation rows, SourceTermEx extraction rows, train-record
    links) are removed too, so a deleted run leaves no orphaned artifacts.

    Args:
        db (Session): Active DB session.
        run_id (int): ID of the TrainingRun to delete.

    Returns:
        bool: True if a run was deleted, False if it did not exist.
    """
    run = db.get(TrainingRun, run_id)
    if run is None:
        return False
    dataset_links = db.exec(
        select(TrainingRunDatasetLink).where(
            TrainingRunDatasetLink.training_run_id == run_id
        )
    ).all()
    for link in dataset_links:
        db.delete(link)
    db.flush()
    if run.model_id is not None:
        model = db.get(Model, run.model_id)
        if model is not None:
            links = db.exec(
                select(ModelTrainRecordLink).where(
                    ModelTrainRecordLink.model_id == model.id
                )
            ).all()
            for link in links:
                db.delete(link)
            db.flush()
            # ORM cascade removes the model's Evaluation + SourceTermEx rows.
            db.delete(model)
    db.delete(run)
    db.commit()
    return True
