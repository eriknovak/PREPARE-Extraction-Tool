import requests
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlmodel import Session, func, select

from app.core.database import Dataset, User, engine, get_session
from app.core.settings import settings
from app.interfaces import Entity, LabelsInput, NERRequest
from app.library.record_processing import auto_link_entities_for_record, link_dates_for_record
from app.models_db import (
    ExtractionJob,
    Model,
    Record,
    SourceTerm,
    SourceTermEx,
    TrainingRun,
)
from app.routes.v1.auth import get_current_user
from app.schemas import (
    ActiveModelResponse,
    ExtractionJobStartResponse,
    ExtractionJobStatusResponse,
    FullStatsRequest,
    FullStatsResponse,
    GLiNERTrainingRequest,
    MessageOutput,
    LabelErrorAnalysis,
    ModelSummary,
    ModelsOutput,
    RunErrorAnalysisResponse,
    RunEvaluationResponse,
    SetActiveModelRequest,
    TrainingMetricPoint,
    TrainingRunsOutput,
    TrainingRunSummary,
    TrainingRunUpdate,
    TrainingStartResponse,
    create_pagination_metadata,
)
import app.services.extraction_lock as extraction_lock
from app.services import (
    bioner_client,
    evaluation_service,
    gliner_data_service,
    training_service,
)
from app.services.websocket_manager import manager

router = APIRouter(tags=["BioNER"])


@router.post("/extract", response_model=List[Entity])
def extract_entities(
    request: NERRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Extract named entities from medical text using the BioNER service.
    """

    try:
        response = requests.post(
            f"{settings.EXTRACT_HOST}/ner", json=request.dict(), timeout=300
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Extract service unavailable",
        )

@router.post("/{dataset_id}/records/{record_id}/extract", response_model=MessageOutput)
def extract_entities_from_record(
    dataset_id: int,
    record_id: int,
    labels: LabelsInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    # Verify dataset ownership
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )
    if dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this dataset",
        )

    statement = (
        select(Record)
        .where(Record.id == record_id)
        .where(Record.dataset_id == dataset_id)
    )
    record = db.exec(statement).one_or_none()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Record not found in this dataset",
        )

    if record.reviewed:
        return MessageOutput(
            message=f"Record {record_id} is reviewed; extraction skipped"
        )

    # Activate the globally selected model (or default) and resolve its Model id
    # up front so it can be used in the already-extracted check below.
    model_id = resolve_active_model(db)

    already_extracted_terms = db.exec(
        select(SourceTermEx)
        .where(SourceTermEx.record_id == record_id)
        .where(SourceTermEx.model_id == model_id)
    ).all()

    if already_extracted_terms:
        current_auto_terms = db.exec(
            select(SourceTerm)
            .where(SourceTerm.record_id == record_id)
            .where(SourceTerm.automatically_extracted == True)  # noqa: E712
        ).all()

        st_keys = {
            (t.value, t.label, t.start_position, t.end_position)
            for t in current_auto_terms
        }

        stex_keys = {
            (t.value, t.label, t.start_position, t.end_position)
            for t in already_extracted_terms
        }

        if st_keys != stex_keys:
            for term in current_auto_terms:
                db.delete(term)
            db.flush()

            existing_source_term_keys = {
                (t.value, t.label, t.start_position, t.end_position)
                for t in db.exec(
                    select(SourceTerm).where(SourceTerm.record_id == record_id)
                ).all()
            }

            new_terms = []
            for ex in already_extracted_terms:
                key = (ex.value, ex.label, ex.start_position, ex.end_position)

                if key in existing_source_term_keys:
                    continue

                existing_source_term_keys.add(key)
                new_terms.append(
                    SourceTerm(
                        record_id=record_id,
                        value=ex.value,
                        label=ex.label,
                        start_position=ex.start_position,
                        end_position=ex.end_position,
                        score=ex.score,
                        automatically_extracted=True,
                    )
                )

            if new_terms:
                db.add_all(new_terms)
                db.flush()
                link_dates_for_record(db, record, dataset)

            db.commit()

            return MessageOutput(
                message=f"Record {record_id} was already extracted with this model; SourceTerms were restored from SourceTermEx."
            )

        return MessageOutput(
            message=f"Record {record_id} was already extracted with this model; extraction skipped."
        )

    # Delete only automatically extracted SourceTerms for this record
    auto_terms = db.exec(
        select(SourceTerm)
        .where(SourceTerm.record_id == record_id)
        .where(SourceTerm.automatically_extracted == True)  # noqa: E712
    ).all()

    for term in auto_terms:
        db.delete(term)
    db.flush()

    request_data = {"medical_text": record.text, "labels": labels.labels}

    try:
        response = requests.post(
            f"{settings.EXTRACT_HOST}/ner", json=request_data, timeout=300
        )
        response.raise_for_status()
        entities = response.json()
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Extraction service unavailable",
        )

    # Keep manually created/edited SourceTerms and avoid duplicating them
    existing_keys = {
        (t.value, t.label, t.start_position, t.end_position)
        for t in db.exec(
            select(SourceTerm).where(SourceTerm.record_id == record_id)
        ).all()
    }

    seen_in_response = set()
    new_terms: List[SourceTerm] = []
    ex_terms: List[SourceTermEx] = []

    for entity in entities:
        key = (
            entity["text"],
            entity["label"],
            entity["start"],
            entity["end"]
        )

        if key in seen_in_response:
            continue
        seen_in_response.add(key)
        ex_terms.append(
            SourceTermEx(
                record_id=record_id,
                value=entity["text"],
                label=entity["label"],
                start_position=entity["start"],
                end_position=entity["end"],
                score=entity.get("score"),
                model_id=model_id,
            )
        )

        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_terms.append(
            SourceTerm(
                record_id=record_id,
                value=entity["text"],
                label=entity["label"],
                start_position=entity["start"],
                end_position=entity["end"],
                score=entity.get("score"),
                automatically_extracted=True,
            )
        )

    if new_terms:
        db.add_all(new_terms)
        db.flush()
        link_dates_for_record(db, record, dataset)
        auto_link_entities_for_record(db, record, dataset)

    if ex_terms:
        db.add_all(ex_terms)
        db.flush()

    db.commit()

    return MessageOutput(
        message=f"Extracted and saved {len(new_terms)} entities from record {record_id}"
    )


@router.post("/{dataset_id}/records/extract", response_model=ExtractionJobStartResponse)
def extract_entities_from_records(
    dataset_id: int,
    labels: LabelsInput,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """
    Kick off extraction for every unreviewed record in the dataset.

    Returns immediately with a job id; progress can be polled via the status endpoint.
    """
    # Verify dataset ownership
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )
    if dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this dataset",
        )

    active_job = db.exec(
        select(ExtractionJob)
        .where(ExtractionJob.dataset_id == dataset_id)
        .where(
            (ExtractionJob.status == "pending") | (ExtractionJob.status == "running")
        )
    ).first()
    if active_job is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An extraction job is already running for this dataset",
        )

    records = db.exec(select(Record).where(Record.dataset_id == dataset_id)).all()
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No records found for this dataset",
        )
    # TODO
    records_to_process = [r for r in records if not r.reviewed]

    # Activate the globally selected model (or default) and resolve its Model id.
    model_id = resolve_active_model(db)

    # check if a job for this dataset and model already exists and is currently "used"
    existing_job = db.exec(
        select(ExtractionJob)
        .where(
            ExtractionJob.dataset_id == dataset_id,
            ExtractionJob.model_id == model_id,
            ExtractionJob.currently_used == True  # noqa: E712
        )
        .order_by(ExtractionJob.created_at.desc())
    ).first()

    if existing_job is None:
        # First extraction for this model on this dataset:
        # delete automatically extracted SourceTerms for unreviewed records only
        unreviewed_record_ids = [r.id for r in records_to_process]

        if unreviewed_record_ids:
            source_terms_to_delete = db.exec(
                select(SourceTerm)
                .where(SourceTerm.record_id.in_(unreviewed_record_ids))
                .where(SourceTerm.automatically_extracted == True)  # noqa: E712
            ).all()
            for st in source_terms_to_delete:
                db.delete(st)

            db.commit()

    # set current job to False, to set new job to True
    currently_used_job = db.exec(
        select(ExtractionJob)
        .where(ExtractionJob.currently_used == True)  # noqa: E712
        .order_by(ExtractionJob.created_at.desc())
    ).first()
    if currently_used_job is not None:
        currently_used_job.currently_used = False
        db.add(currently_used_job)
        db.commit()

    total = len(records_to_process)
    job = ExtractionJob(
        dataset_id=dataset_id,
        model_id=model_id,
        total=total,
        completed=0,
        status="pending",
        currently_used=True
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    if total == 0:
        job.status = "completed"
        job.updated_at = datetime.now(timezone.utc)
        db.add(job)
        db.commit()

        return ExtractionJobStartResponse(
            job_id=job.id,
            dataset_id=dataset_id,
            total=job.total,
            status=job.status,
        )

    background_tasks.add_task(
        run_dataset_extraction_job,
        job_id=job.id,
        dataset_id=dataset_id,
        labels=labels.labels,
        model_id=model_id,
    )

    return ExtractionJobStartResponse(
        job_id=job.id,
        dataset_id=dataset_id,
        total=total,
        status=job.status,
    )


@router.get(
    "/{dataset_id}/records/extract/active",
    response_model=Optional[ExtractionJobStatusResponse],
)
def get_active_extraction_job(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Return the latest pending/running extraction job for the dataset, or null if none."""
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    if dataset.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this dataset")

    job = db.exec(
        select(ExtractionJob)
        .where(ExtractionJob.dataset_id == dataset_id)
        .where(
            (ExtractionJob.status == "pending") | (ExtractionJob.status == "running")
        )
        .order_by(ExtractionJob.created_at.desc())
    ).first()

    if job is None:
        return None

    return ExtractionJobStatusResponse(
        job_id=job.id,
        dataset_id=job.dataset_id,
        total=job.total,
        completed=job.completed,
        status=job.status,
        error_message=job.error_message,
    )


@router.get(
    "/{dataset_id}/records/extract/{job_id}/status",
    response_model=ExtractionJobStatusResponse,
)
def get_extraction_job_status(
    dataset_id: int,
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Return progress for a dataset extraction job."""

    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )
    if dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this dataset",
        )

    job = db.get(ExtractionJob, job_id)
    if job is None or job.dataset_id != dataset_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Extraction job not found for this dataset",
        )

    return ExtractionJobStatusResponse(
        job_id=job.id,
        dataset_id=job.dataset_id,
        total=job.total,
        completed=job.completed,
        status=job.status,
        error_message=job.error_message,
    )


@router.post(
    "/{dataset_id}/records/extract/{job_id}/cancel",
    response_model=MessageOutput,
)
def cancel_extraction_job(
    dataset_id: int,
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Request cancellation of an extraction job. Already-processed records remain."""

    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )
    if dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this dataset",
        )

    job = db.get(ExtractionJob, job_id)
    if job is None or job.dataset_id != dataset_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Extraction job not found for this dataset",
        )

    if job.status in {"completed", "failed", "cancelled"}:
        return MessageOutput(message=f"Job already {job.status}")

    job.status = "cancelled"
    job.updated_at = datetime.now(timezone.utc)
    db.add(job)
    db.commit()

    return MessageOutput(message="Cancellation requested")


def run_dataset_extraction_job(job_id: int, dataset_id: int, labels: List[str], model_id: int):
    """Background task that extracts entities for each unreviewed record."""

    with Session(engine) as session:
        job = session.get(ExtractionJob, job_id)
        if job is None:
            return

        dataset = session.get(Dataset, dataset_id)

        if job.status == "cancelled":
            return

        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)
        session.add(job)
        session.commit()

        records = session.exec(
            select(Record).where(Record.dataset_id == dataset_id)
        ).all()

        # Skip reviewed records and records already containing extracted terms with current model
        unreviewed_records = [r for r in records if not r.reviewed]
        processed_records = []
        records_to_process: List[Record] = []

        for record in unreviewed_records:
            # if the model already processed this Record, skip it
            has_extraction_for_model = session.exec(
                select(SourceTermEx.id)
                .where(SourceTermEx.record_id == record.id)
                .where(SourceTermEx.model_id == model_id)
            ).first()

            if has_extraction_for_model:
                processed_records.append(record)
            else:
                records_to_process.append(record)

        job.total = len(unreviewed_records)
        job.completed = len(processed_records)
        job.updated_at = datetime.now(timezone.utc)
        session.add(job)
        session.commit()

        for record in records_to_process:
            session.refresh(job)
            if job.status == "cancelled":
                job.updated_at = datetime.now(timezone.utc)
                session.add(job)
                session.commit()
                return

            request_data = {"medical_text": record.text, "labels": labels}
            try:
                response = requests.post(
                    f"{settings.EXTRACT_HOST}/ner", json=request_data, timeout=300
                )
                response.raise_for_status()
                entities = response.json()
            except requests.RequestException as exc:
                job.status = "failed"
                job.error_message = str(exc)
                job.updated_at = datetime.now(timezone.utc)
                session.add(job)
                session.commit()
                return

            existing_keys = {
                (t.value, t.label, t.start_position, t.end_position)
                for t in session.exec(
                    select(SourceTerm).where(SourceTerm.record_id == record.id)
                ).all()
            }
            
            seen_in_response = set()

            new_terms: List[SourceTerm] = []
            ex_terms: List[SourceTermEx] = []
            for entity in entities:
                key = (
                    entity["text"],
                    entity["label"],
                    entity["start"],
                    entity["end"]
                )

                if key in seen_in_response:
                    continue
                seen_in_response.add(key)
                ex_terms.append(
                    SourceTermEx(
                        record_id=record.id,
                        value=entity["text"],
                        label=entity["label"],
                        start_position=entity["start"],
                        end_position=entity["end"],
                        score=entity.get("score"),
                        model_id=model_id
                    )
                )

                if key in existing_keys:
                    continue
                existing_keys.add(key)
                new_terms.append(
                    SourceTerm(
                        record_id=record.id,
                        value=entity["text"],
                        label=entity["label"],
                        start_position=entity["start"],
                        end_position=entity["end"],
                        score=entity.get("score"),
                        automatically_extracted=True,
                    )
                )
                
            if new_terms:
                session.add_all(new_terms)
                session.flush()
                link_dates_for_record(session, record, dataset)
                auto_link_entities_for_record(session, record, dataset)
            if ex_terms:
                session.add_all(ex_terms)
                session.flush()

            job.completed += 1
            job.updated_at = datetime.now(timezone.utc)
            session.add(job)
            session.commit()

        job.status = "completed"
        job.updated_at = datetime.now(timezone.utc)
        session.add(job)
        session.commit()


# ================================================
# NER model routes
# ================================================

def get_or_create_model(metadata: dict, db: Session):
    statement = select(Model).where(
        Model.name == metadata["name"],
        Model.version == metadata["version"]
    )
    existing = db.exec(statement).first()
    if existing:
        return existing
    
    new_model = Model(
        name=metadata["name"],
        version=metadata["version"]
    )

    db.add(new_model)
    db.commit()
    db.refresh(new_model)

    return new_model


def _model_summary(db: Session, model: Model) -> ModelSummary:
    """Build a ModelSummary, including the model's overall macro-F1 if evaluated."""
    per_label = evaluation_service.get_per_label(db, model.id)
    score = evaluation_service.compute_macro_f1(per_label)
    return ModelSummary(
        id=model.id,
        name=model.name,
        version=model.version,
        base_model=model.base_model,
        path=model.path,
        dataset_id=model.dataset_id,
        created_at=model.created_at,
        score=score,
    )


def resolve_active_model(db: Session) -> int:
    """Return the Model id to record extracted terms under, using the GLOBAL
    active model. Ensures bioner is on the right model and returns its id.

    With a global selection there is no per-call swap in the hot path during a
    job (the model was activated when it was selected), but we still confirm the
    target here so a fresh worker / restart converges. Raises 503 if bioner is
    unreachable when a default lookup is required.
    """
    model_db = training_service.get_global_active_model(db)
    if model_db is not None and model_db.path:
        return model_db.id

    # Default model: record under the default's metadata row.
    try:
        info = requests.get(f"{settings.EXTRACT_HOST}/model/info", timeout=30)
        info.raise_for_status()
        metadata = info.json()["model"]
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Extraction service unavailable",
        )
    return get_or_create_model(metadata, db).id


# ================================================
# NER model selection routes
# ================================================


@router.get("/models", response_model=ModelsOutput)
def list_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """List the caller's trained models (those with an artifact path) for selection."""
    models = db.exec(
        select(Model)
        .join(Dataset, Dataset.id == Model.dataset_id)
        .where(Dataset.user_id == current_user.id)
        .where(Model.path.is_not(None))
        .order_by(Model.created_at.desc())
    ).all()
    return ModelsOutput(models=[_model_summary(db, m) for m in models])


def _activate_on_bioner(model_path: Optional[str]) -> None:
    """Hot-swap bioner to the given model path (None = revert to launch default)."""
    try:
        resp = requests.post(
            f"{settings.EXTRACT_HOST}/model/activate",
            json={"model": model_path},
            timeout=300,
        )
        resp.raise_for_status()
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Extraction service unavailable",
        )


@router.get("/active-model", response_model=ActiveModelResponse)
def get_active_model(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Return the GLOBAL active extraction model (null = bioner default)."""
    model = training_service.get_global_active_model(db)
    active = _model_summary(db, model) if model is not None else None
    return ActiveModelResponse(active_model=active)


@router.post("/active-model", response_model=ActiveModelResponse)
def set_active_model(
    payload: SetActiveModelRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Set/clear the GLOBAL active extraction model and hot-swap bioner.

    Blocked (409) while any extraction job is active instance-wide so an in-flight
    job's pinned model can't be undermined.
    """
    if extraction_lock.any_extraction_job_active(db):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot change the model while an extraction job is running",
        )

    if payload.model_id is None:
        _activate_on_bioner(None)
        training_service.set_global_active_model(db, None)
        return ActiveModelResponse(active_model=None)

    model = db.get(Model, payload.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if model.path is None:
        raise HTTPException(
            status_code=400,
            detail="Model has no trained artifact to use for extraction",
        )
    _activate_on_bioner(model.path)
    training_service.set_global_active_model(db, model.id)
    return ActiveModelResponse(active_model=_model_summary(db, model))


# ================================================
# Training / monitoring routes
# ================================================


@router.post("/training/start", response_model=TrainingStartResponse)
def start_training(
    req: GLiNERTrainingRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    # ``req.dataset_ids`` is resolved + de-duplicated by the schema validator.
    train_ids = req.dataset_ids
    eval_ids = req.eval_dataset_ids

    # Verify the caller owns every selected dataset (train + eval).
    for dsid in [*train_ids, *eval_ids]:
        dataset = db.get(Dataset, dsid)
        if dataset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dsid} not found",
            )
        if dataset.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not authorized to access dataset {dsid}",
            )

    # Reject a second concurrent run touching any of the training datasets.
    if training_service.has_active_run_for_datasets(db, train_ids):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A training run is already active for one of these datasets",
        )

    train_stats_snapshot = {
        "train_dataset_ids": train_ids,
        "eval_dataset_ids": eval_ids,
        **_compute_dataset_stats(db, train_ids),
        "val_ratio": req.val_ratio,
    }
    run = training_service.create_run(
        db,
        dataset_ids=train_ids,
        base_model=req.base_model,
        labels=req.labels,
        val_ratio=req.val_ratio,
        eval_dataset_ids=eval_ids,
        train_stats=train_stats_snapshot,
    )
    try:
        training_data = gliner_data_service.load_reviewed_training_data(
            db, train_ids, req.labels
        )
        # Separate eval datasets override the held-out split when provided.
        eval_data = (
            gliner_data_service.load_reviewed_training_data(db, eval_ids, req.labels)
            if eval_ids
            else []
        )
        bioner_client.start_training(
            {
                "run_id": run.id,
                "base_model": req.base_model,
                "training_data": training_data,
                "eval_data": eval_data,
                "val_ratio": req.val_ratio,
                "num_epochs": req.num_epochs,
                "learning_rate": req.learning_rate,
                "train_batch_size": req.train_batch_size,
            }
        )
    except Exception as exc:  # trainer unreachable -> mark failed, but still return run
        training_service.fail_run(db, run.id, f"failed to start trainer: {exc}")
    return TrainingStartResponse(run_id=run.id)


@router.post("/training/stop/{run_id}", response_model=MessageOutput)
def stop_training(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    training_service.stop_run(db, run_id)
    try:
        bioner_client.stop_training(run_id)
    except Exception:
        pass
    return MessageOutput(message="stopped")


def _compute_dataset_stats(db: Session, dataset_ids: list) -> dict:
    """Compute aggregated stats across the given dataset IDs.

    Returns a dict with ``record_count``, ``term_count``, and
    ``label_distribution`` (label -> count mapping).
    """
    record_count = db.exec(
        select(func.count(Record.id)).where(Record.dataset_id.in_(dataset_ids))
    ).one()
    term_count = db.exec(
        select(func.count(SourceTerm.id))
        .join(Record, Record.id == SourceTerm.record_id)
        .where(Record.dataset_id.in_(dataset_ids))
    ).one()
    rows = db.exec(
        select(SourceTerm.label, func.count(SourceTerm.id))
        .join(Record, Record.id == SourceTerm.record_id)
        .where(Record.dataset_id.in_(dataset_ids))
        .group_by(SourceTerm.label)
    ).all()
    return {
        "record_count": record_count,
        "term_count": term_count,
        "label_distribution": {label: count for label, count in rows},
    }


@router.get("/datasets/{dataset_id}/full-stats", response_model=FullStatsResponse)
def full_stats(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    total_records = db.exec(
        select(func.count(Record.id)).where(Record.dataset_id == dataset_id)
    ).one()
    total_terms = db.exec(
        select(func.count(SourceTerm.id))
        .join(Record, Record.id == SourceTerm.record_id)
        .where(Record.dataset_id == dataset_id)
    ).one()
    rows = db.exec(
        select(SourceTerm.label, func.count(SourceTerm.id))
        .join(Record, Record.id == SourceTerm.record_id)
        .where(Record.dataset_id == dataset_id)
        .group_by(SourceTerm.label)
    ).all()
    return FullStatsResponse(
        totalRecords=total_records,
        totalTerms=total_terms,
        labelDistribution={label: count for label, count in rows},
    )


@router.post("/datasets/full-stats", response_model=FullStatsResponse)
def full_stats_multi(
    req: FullStatsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Aggregate record/term counts and label distribution across datasets."""
    stats = _compute_dataset_stats(db, req.dataset_ids)
    return FullStatsResponse(
        totalRecords=stats["record_count"],
        totalTerms=stats["term_count"],
        labelDistribution=stats["label_distribution"],
    )


def _run_summary(db: Session, run: TrainingRun) -> TrainingRunSummary:
    """Build an enriched run summary, including artifact path and macro-F1 score."""
    path = None
    score = None
    if run.model_id is not None:
        model = db.get(Model, run.model_id)
        if model is not None:
            path = model.path
        per_label = evaluation_service.get_per_label(db, run.model_id)
        score = evaluation_service.compute_macro_f1(per_label)
    return TrainingRunSummary(
        run_id=run.id,
        status=run.status,
        name=run.name,
        base_model=run.base_model,
        labels=run.labels or [],
        val_ratio=run.val_ratio,
        created_at=run.created_at,
        error_message=run.error_message,
        path=path,
        model_id=run.model_id,
        score=score,
        preferred=run.preferred,
    )


def _get_owned_run(db: Session, run_id: int, current_user: User) -> TrainingRun:
    """Fetch a run and verify the caller owns its dataset, else raise 404/403."""
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Run not found"
        )
    dataset = db.get(Dataset, run.dataset_id)
    if dataset is not None and dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this run",
        )
    return run


@router.get("/datasets/{dataset_id}/runs", response_model=TrainingRunsOutput)
def list_runs(
    dataset_id: int,
    page: int = Query(1, ge=1, description="Page number (newest first)"),
    limit: int = Query(20, ge=1, le=100, description="Runs per page"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Return a dataset's training runs, newest first, paginated."""
    total = db.exec(
        select(func.count(TrainingRun.id)).where(
            TrainingRun.dataset_id == dataset_id
        )
    ).one()
    offset = (page - 1) * limit
    runs = db.exec(
        select(TrainingRun)
        .where(TrainingRun.dataset_id == dataset_id)
        .order_by(TrainingRun.id.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return TrainingRunsOutput(
        runs=[_run_summary(db, r) for r in runs],
        pagination=create_pagination_metadata(total, limit, offset),
    )


@router.patch("/runs/{run_id}", response_model=TrainingRunSummary)
def update_run(
    run_id: int,
    payload: TrainingRunUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Rename a run and/or mark it as the dataset's preferred run."""
    _get_owned_run(db, run_id, current_user)
    run = training_service.update_run(
        db, run_id, name=payload.name, preferred=payload.preferred
    )
    return _run_summary(db, run)


@router.delete("/runs/{run_id}", response_model=MessageOutput)
def delete_run(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Delete a run and its dependent metrics/model/evaluation rows."""
    _get_owned_run(db, run_id, current_user)
    training_service.delete_run(db, run_id)
    return MessageOutput(message="deleted")


def _scores_only(per_label: dict) -> dict:
    """Drop the heavy per-label ``examples`` arrays, keeping scalar scores/counts.

    Used by the score-oriented endpoints (single-run bars, heatmap) so they don't
    ship example-error text; the full data is served by the error-analysis route.
    """
    return {
        label: {k: v for k, v in metrics.items() if k != "examples"}
        if isinstance(metrics, dict)
        else metrics
        for label, metrics in per_label.items()
    }


@router.get("/runs/{run_id}/evaluation", response_model=RunEvaluationResponse)
def run_evaluation(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    run = db.get(TrainingRun, run_id)
    per_label = {}
    if run is not None and run.model_id is not None:
        per_label = _scores_only(evaluation_service.get_per_label(db, run.model_id))
    return RunEvaluationResponse(run_id=run_id, per_label=per_label)


@router.get("/runs/{run_id}/error-analysis", response_model=RunErrorAnalysisResponse)
def run_error_analysis(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Return per-label error analysis for a run: precision/recall, false-positive
    and false-negative counts, and a bounded sample of concrete example errors.

    The data is read from the run model's stored per-label evaluation JSON. Runs
    trained before error analysis was added have no such data; for those
    ``available`` is False and ``per_label`` is empty.
    """
    run = _get_owned_run(db, run_id, current_user)

    per_label_raw = (
        evaluation_service.get_per_label(db, run.model_id) if run.model_id else {}
    )
    # Newer runs always carry fp/fn/examples; their presence marks the data as
    # available. Older runs only have F1/precision/recall scores.
    available = any(
        isinstance(m, dict) and ("fp" in m or "examples" in m)
        for m in per_label_raw.values()
    )

    per_label = {}
    if available:
        for label, m in per_label_raw.items():
            if not isinstance(m, dict):
                continue
            per_label[label] = LabelErrorAnalysis(
                precision=m.get("precision"),
                recall=m.get("recall"),
                fp=m.get("fp"),
                fn=m.get("fn"),
                examples=m.get("examples") or [],
            )

    return RunErrorAnalysisResponse(
        run_id=run_id, available=available, per_label=per_label
    )


@router.get("/runs/{run_id}/metrics", response_model=List[TrainingMetricPoint])
def run_metrics(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Return a run's per-epoch loss curve, ordered by epoch."""
    metrics = training_service.get_run_metrics(db, run_id)
    return [TrainingMetricPoint(epoch=m.epoch, loss=m.loss) for m in metrics]


@router.get("/datasets/{dataset_id}/runs/evaluations")
def dataset_runs_evaluations(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    out = []

    # Prepend the default extraction model as a baseline (run_id 0, not a real
    # run) so trained runs can be compared against the starting point. Only
    # included once it has been evaluated on this dataset's eval split.
    baseline = training_service.get_baseline_model(db, dataset_id)
    if baseline is not None:
        baseline_per_label = _scores_only(
            evaluation_service.get_per_label(db, baseline.id)
        )
        if baseline_per_label:
            out.append(
                {
                    "run_id": 0,
                    "name": "Base model",
                    "is_baseline": True,
                    "per_label": baseline_per_label,
                }
            )

    runs = db.exec(
        select(TrainingRun).where(TrainingRun.dataset_id == dataset_id)
    ).all()
    for r in runs:
        per_label = (
            _scores_only(evaluation_service.get_per_label(db, r.model_id))
            if r.model_id
            else {}
        )
        out.append({"run_id": r.id, "per_label": per_label})
    return out


@router.websocket("/ws/training")
async def ws_training(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)