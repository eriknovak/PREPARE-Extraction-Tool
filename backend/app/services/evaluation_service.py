"""Persist and read per-label NER evaluation results."""

from typing import Any, Dict

from sqlmodel import Session, select

from app.models_db import Evaluation


def store_evaluation(
    db: Session,
    *,
    model_id: int,
    dataset_id: int,
    per_label: Dict[str, Dict[str, float]],
) -> None:
    """Replace evaluation rows for a model with one row per label.

    Args:
        db (Session): Active DB session.
        model_id (int): Model the evaluation belongs to.
        dataset_id (int): Dataset the model was evaluated on.
        per_label (Dict[str, Dict[str, float]]): Label -> metric mapping. Each
            metric mapping is stored verbatim as flexible JSON (e.g. exact_f1,
            relaxed_f1, precision, recall).
    """
    existing = db.exec(select(Evaluation).where(Evaluation.model_id == model_id)).all()
    for row in existing:
        db.delete(row)
    db.flush()
    for label, score in per_label.items():
        db.add(Evaluation(label=label, score=score, dataset_id=dataset_id, model_id=model_id))
    db.commit()


def get_per_label(db: Session, model_id: int) -> Dict[str, Dict[str, Any]]:
    """Return {label: score} for a model's evaluation.

    Args:
        db (Session): Active DB session.
        model_id (int): Model id.

    Returns:
        Dict[str, Dict[str, Any]]: Label -> metric mapping.
    """
    rows = db.exec(select(Evaluation).where(Evaluation.model_id == model_id)).all()
    return {row.label: row.score for row in rows}
