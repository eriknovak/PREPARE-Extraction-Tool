# Monitoring-view Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the GLiNER training pipeline + monitoring dashboard from `Monitoring-view` into the codebase, reconciled onto `ninakokalj/main`'s database schema.

**Architecture:** A dedicated `training_run` table owns the training lifecycle and, on success, produces a `Model` row (ninakokalj's extended artifact table). Per-epoch `loss` is stored in `training_metric`; final per-label evaluation is stored in ninakokalj's `Evaluation` table as flexible JSON (`exact_f1`, `relaxed_f1`, precision, recall). Training/monitoring endpoints live under `/bioner`; logic lives in services, with `bioner.py` as a thin route layer.

**Tech Stack:** FastAPI, SQLModel, PostgreSQL, Alembic, pytest (+ FastAPI TestClient), websockets, GLiNER (sentence-transformers).

**Reference spec:** `docs/superpowers/specs/2026-06-18-monitoring-view-integration-design.md`

**Source branch for ports:** `e3-jsi/Monitoring-view` (read with `git show e3-jsi/Monitoring-view:<path>`).

---

## File Structure

**Backend — new:**
- `backend/app/library/ner_metrics.py` — exact/relaxed/overlap/bertscore F1 (port).
- `backend/app/services/evaluation_service.py` — compute + store `Evaluation` rows, build `per_label` responses.
- `backend/app/services/training_service.py` — run lifecycle state machine + event handling.
- `backend/app/services/websocket_manager.py` — live broadcast (port).
- `backend/app/services/bioner_client.py` — trainer client (port).

**Backend — modified:**
- `backend/app/models_db.py` — extend `Model`; add `TrainingRun`, `TrainingMetric`.
- `backend/app/schemas.py` — `GLiNERTrainingRequest` + monitoring response schemas.
- `backend/app/routes/v1/bioner.py` — add training/monitoring routes (thin).
- `backend/app/alembic/versions/` — one additive migration.

**bioner service — new (port):**
- `bioner/app/library/ner_metrics.py`
- `bioner/app/training/{__init__,trainer,gliner_trainer,job_manager,callbacks}.py`
- `bioner/app/routes/v1/training.py` (or merge into `bioner/app/main.py`, matching base layout)

**Tests — new:**
- `backend/tests/test_ner_metrics.py`
- `backend/tests/test_evaluation_service.py`
- `backend/tests/test_training_service.py`
- `backend/tests/test_training_routes.py`

**Explicitly NOT ported (deferred bucket C/D):** `UserModelPreference` + model-selection routes, `ModelArtifact`, `TrainingEvaluation`, `SourceTermLink`/label-linking, storybook, model-settings UI, the dead `monitoring.py` router.

---

## Phase 1 — Schema + Migration

### Task 1: Extend `Model` and add training tables

**Files:**
- Modify: `backend/app/models_db.py` (`Model` class ~line 251; add new classes after `Evaluation` ~line 307)

- [ ] **Step 1: Add `Dict, Any` import if missing**

At top of `backend/app/models_db.py`, ensure:
```python
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, JSON
```

- [ ] **Step 2: Extend the `Model` class**

Add these fields to `class Model` (keep existing `id`, `name`, `version`, relationships):
```python
    base_model: Optional[str] = Field(default=None)
    path: Optional[str] = Field(default=None)
    dataset_id: Optional[int] = Field(
        default=None, foreign_key="dataset.id", ondelete="SET NULL"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # One-to-one back to the training run that produced this model
    training_run: Optional["TrainingRun"] = Relationship(back_populates="model")
```

- [ ] **Step 3: Add `TrainingRun` and `TrainingMetric` after `Evaluation`**

```python
class TrainingRun(SQLModel, table=True):
    """A GLiNER training run and its lifecycle. Produces a Model on success."""

    __tablename__ = "training_run"

    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(
        foreign_key="dataset.id", ondelete="CASCADE", nullable=False, index=True
    )
    base_model: str
    labels: List[str] = Field(sa_column=Column(JSON))
    val_ratio: float = Field(default=0.0)
    status: str = Field(default="pending", index=True)  # pending|running|completed|failed|stopped
    error_message: Optional[str] = Field(default=None)
    model_id: Optional[int] = Field(
        default=None, foreign_key="model.id", ondelete="SET NULL"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    metrics: list["TrainingMetric"] = Relationship(
        back_populates="run",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    model: Optional["Model"] = Relationship(back_populates="training_run")


class TrainingMetric(SQLModel, table=True):
    """Per-epoch training metric (loss curve)."""

    __tablename__ = "training_metric"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(
        foreign_key="training_run.id", ondelete="CASCADE", nullable=False, index=True
    )
    epoch: int
    loss: Optional[float] = Field(default=None)

    run: Optional["TrainingRun"] = Relationship(back_populates="metrics")
```

- [ ] **Step 4: Verify models import cleanly**

Run: `cd backend && python -c "import app.models_db"`
Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add backend/app/models_db.py
git commit -m "feat(db): extend Model and add training_run/training_metric tables"
```

### Task 2: Alembic migration

**Files:**
- Create: `backend/app/alembic/versions/<rev>_add_training_run_and_model_fields.py`

- [ ] **Step 1: Autogenerate against a running DB**

Run (from `backend/`, DB up):
```bash
alembic revision --autogenerate --rev-id 010 -m "add training_run and model training fields"
```
Expected: a new file under `app/alembic/versions/`.

- [ ] **Step 2: Review the generated migration**

Confirm `upgrade()` does exactly:
- `op.add_column("model", base_model/path/dataset_id/created_at)` + FK `model.dataset_id → dataset.id` (`ondelete="SET NULL"`).
- `op.create_table("training_run", ...)` with FKs to `dataset` (CASCADE) and `model` (SET NULL) + indexes on `dataset_id`, `status`.
- `op.create_table("training_metric", ...)` with FK to `training_run` (CASCADE) + index on `run_id`.

Confirm it does **not** touch `evaluation` / `model_train_record_link` (already created by ninakokalj). Delete any spurious ops. Ensure `downgrade()` drops them in reverse.

- [ ] **Step 3: Round-trip the migration**

Run: `alembic upgrade head && alembic downgrade -1 && alembic upgrade head`
Expected: all succeed, no errors.

- [ ] **Step 4: Commit**

```bash
git add backend/app/alembic/versions/010_*.py
git commit -m "feat(db): migration for training tables and Model fields"
```

---

## Phase 2 — Metrics + Evaluation service

### Task 3: Port `ner_metrics.py` (backend) with tests

**Files:**
- Create: `backend/app/library/ner_metrics.py`
- Test: `backend/tests/test_ner_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_ner_metrics.py
from app.interfaces import Entity
from app.library.ner_metrics import NERMetrics


def _ent(text, label, start, end):
    return Entity(text=text, label=label, start=start, end=end, score=1.0)


def test_exact_f1_perfect_match():
    true = [[_ent("aspirin", "Drug", 0, 7)]]
    pred = [[_ent("aspirin", "Drug", 0, 7)]]
    p, r, f1 = NERMetrics(["exact"]).evaluate_ner_performance(true, pred, "exact")
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_relaxed_matches_partial_span_exact_does_not():
    true = [[_ent("aspirin 100mg", "Drug", 0, 13)]]
    pred = [[_ent("aspirin", "Drug", 0, 7)]]
    _, _, exact_f1 = NERMetrics(["exact"]).evaluate_ner_performance(true, pred, "exact")
    _, _, relaxed_f1 = NERMetrics(["relaxed"]).evaluate_ner_performance(true, pred, "relaxed")
    assert exact_f1 == 0.0
    assert relaxed_f1 == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_ner_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: app.library.ner_metrics`.

- [ ] **Step 3: Port the implementation verbatim**

Run: `git show e3-jsi/Monitoring-view:backend/app/library/ner_metrics.py > backend/app/library/ner_metrics.py`
(It imports `from app.interfaces import Entity`, which already exists with `text/label/start/end/score`. No edits needed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_ner_metrics.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/library/ner_metrics.py backend/tests/test_ner_metrics.py
git commit -m "feat(metrics): port NER exact/relaxed F1 metrics with tests"
```

### Task 4: `evaluation_service` — store & assemble per-label evaluation

**Files:**
- Create: `backend/app/services/evaluation_service.py`
- Test: `backend/tests/test_evaluation_service.py`

**Interface (used by later tasks):**
- `store_evaluation(db, *, model_id: int, dataset_id: int, per_label: Dict[str, Dict[str, float]]) -> None`
  — deletes existing `Evaluation` rows for `model_id`, inserts one row per label with `score=metrics dict`.
- `get_per_label(db, model_id: int) -> Dict[str, Dict[str, Any]]`
  — returns `{label: score}` for a model.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_evaluation_service.py
from app.services import evaluation_service as svc


def test_store_and_get_per_label(session, sample_dataset, sample_model):
    per_label = {
        "Drug": {"exact_f1": 0.8, "relaxed_f1": 0.9, "precision": 0.85, "recall": 0.75},
        "Diagnosis": {"exact_f1": 0.6, "relaxed_f1": 0.7, "precision": 0.6, "recall": 0.6},
    }
    svc.store_evaluation(
        session, model_id=sample_model.id, dataset_id=sample_dataset.id, per_label=per_label
    )
    result = svc.get_per_label(session, sample_model.id)
    assert result["Drug"]["exact_f1"] == 0.8
    assert result["Diagnosis"]["relaxed_f1"] == 0.7


def test_store_evaluation_is_idempotent(session, sample_dataset, sample_model):
    svc.store_evaluation(session, model_id=sample_model.id, dataset_id=sample_dataset.id,
                         per_label={"Drug": {"exact_f1": 0.1}})
    svc.store_evaluation(session, model_id=sample_model.id, dataset_id=sample_dataset.id,
                         per_label={"Drug": {"exact_f1": 0.9}})
    result = svc.get_per_label(session, sample_model.id)
    assert result["Drug"]["exact_f1"] == 0.9  # replaced, not duplicated
```

(Fixtures `session`, `sample_dataset`, `sample_model` are added in Task 5's conftest step; if running this task first, add them now — see Task 5 Step 1.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_evaluation_service.py -v`
Expected: FAIL — module/attr not found.

- [ ] **Step 3: Implement the service**

```python
# backend/app/services/evaluation_service.py
"""Persist and read per-label NER evaluation results."""

from typing import Any, Dict

from sqlmodel import Session, delete, select

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
    db.exec(delete(Evaluation).where(Evaluation.model_id == model_id))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_evaluation_service.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/evaluation_service.py backend/tests/test_evaluation_service.py
git commit -m "feat(eval): evaluation service storing flexible per-label JSON metrics"
```

---

## Phase 3 — Training lifecycle service

### Task 5: `training_service` state machine + event handling

**Files:**
- Create: `backend/app/services/training_service.py`
- Test: `backend/tests/test_training_service.py`
- Modify: `backend/tests/conftest.py` (add fixtures)

**Interface (used by routes in Phase 4):**
- `create_run(db, *, dataset_id, base_model, labels, val_ratio) -> TrainingRun`
- `mark_running(db, run_id) -> None`
- `add_epoch_metric(db, run_id, epoch, loss) -> None`
- `record_evaluation(db, run_id, per_label) -> None` — ensures a `Model`, links it, stores eval via `evaluation_service`.
- `complete_run(db, run_id, output_path, record_ids) -> Model` — ensures `Model` (sets path/base_model/dataset/version), sets `run.model_id`, populates `ModelTrainRecordLink`, `status="completed"`.
- `fail_run(db, run_id, message) -> None`
- `stop_run(db, run_id) -> None`

- [ ] **Step 1: Add shared test fixtures**

In `backend/tests/conftest.py` add (adapt to the existing in-memory/session fixture already used by `test_datasets.py`):
```python
import pytest
from app.models_db import Dataset, Model, Record


@pytest.fixture
def sample_dataset(session, sample_user):
    ds = Dataset(name="ds", labels=["Drug", "Diagnosis"], user_id=sample_user.id)
    session.add(ds); session.commit(); session.refresh(ds)
    return ds


@pytest.fixture
def sample_model(session, sample_dataset):
    m = Model(name="m", version="1", dataset_id=sample_dataset.id)
    session.add(m); session.commit(); session.refresh(m)
    return m


@pytest.fixture
def sample_record(session, sample_dataset):
    rec = Record(dataset_id=sample_dataset.id, text="aspirin 100mg")
    session.add(rec); session.commit(); session.refresh(rec)
    return rec
```
(Inspect the existing `conftest.py` first; reuse its `session`/`sample_user` fixtures and `Record`/`Dataset` required fields — match field names exactly to `models_db.py`.)

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_training_service.py
from app.models_db import Evaluation, ModelTrainRecordLink, TrainingMetric, TrainingRun
from app.services import training_service as svc
from sqlmodel import select


def test_run_lifecycle_to_completed(session, sample_dataset, sample_record):
    run = svc.create_run(session, dataset_id=sample_dataset.id,
                         base_model="urchade/gliner_small-v2.1",
                         labels=["Drug"], val_ratio=0.1)
    assert run.status == "pending"

    svc.mark_running(session, run.id)
    assert session.get(TrainingRun, run.id).status == "running"

    svc.add_epoch_metric(session, run.id, epoch=1, loss=0.5)
    svc.add_epoch_metric(session, run.id, epoch=2, loss=0.3)
    metrics = session.exec(select(TrainingMetric).where(TrainingMetric.run_id == run.id)).all()
    assert [m.loss for m in sorted(metrics, key=lambda x: x.epoch)] == [0.5, 0.3]

    svc.record_evaluation(session, run.id, {"Drug": {"exact_f1": 0.8, "relaxed_f1": 0.9}})
    model = svc.complete_run(session, run.id, output_path="/models/run.pt",
                             record_ids=[sample_record.id])

    refreshed = session.get(TrainingRun, run.id)
    assert refreshed.status == "completed"
    assert refreshed.model_id == model.id
    assert model.path == "/models/run.pt"
    assert model.base_model == "urchade/gliner_small-v2.1"

    evals = session.exec(select(Evaluation).where(Evaluation.model_id == model.id)).all()
    assert {e.label for e in evals} == {"Drug"}
    links = session.exec(
        select(ModelTrainRecordLink).where(ModelTrainRecordLink.model_id == model.id)
    ).all()
    assert {l.record_id for l in links} == {sample_record.id}


def test_fail_run_records_message(session, sample_dataset):
    run = svc.create_run(session, dataset_id=sample_dataset.id, base_model="b",
                         labels=["Drug"], val_ratio=0.0)
    svc.fail_run(session, run.id, "boom")
    refreshed = session.get(TrainingRun, run.id)
    assert refreshed.status == "failed"
    assert refreshed.error_message == "boom"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && pytest tests/test_training_service.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement the service**

```python
# backend/app/services/training_service.py
"""Training run lifecycle: create, progress, evaluate, complete/fail/stop."""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlmodel import Session

from app.models_db import (
    Evaluation,
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
    """Create a TrainingRun in the 'pending' state."""
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
    """Transition a run to 'running'."""
    run = db.get(TrainingRun, run_id)
    if run is None:
        return
    run.status = "running"
    db.add(run)
    db.commit()


def add_epoch_metric(db: Session, run_id: int, epoch: int, loss: Optional[float]) -> None:
    """Append a per-epoch loss point."""
    db.add(TrainingMetric(run_id=run_id, epoch=epoch, loss=loss))
    db.commit()


def _ensure_model(db: Session, run: TrainingRun) -> Model:
    """Return the run's Model, creating and linking one if absent."""
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


def record_evaluation(db: Session, run_id: int, per_label: Dict[str, Dict[str, float]]) -> None:
    """Store final per-label evaluation against the run's Model."""
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
    """Finalize a successful run: set artifact path, link training records."""
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
    """Mark a run failed with an error message."""
    run = db.get(TrainingRun, run_id)
    if run is None:
        return
    run.status = "failed"
    run.error_message = message
    db.add(run)
    db.commit()


def stop_run(db: Session, run_id: int) -> None:
    """Mark a run stopped."""
    run = db.get(TrainingRun, run_id)
    if run is None:
        return
    run.status = "stopped"
    db.add(run)
    db.commit()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_training_service.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/training_service.py backend/tests/test_training_service.py backend/tests/conftest.py
git commit -m "feat(training): training run lifecycle service with tests"
```

### Task 6: Port websocket manager + bioner trainer client

**Files:**
- Create: `backend/app/services/websocket_manager.py`
- Create: `backend/app/services/bioner_client.py`

- [ ] **Step 1: Port both files**

```bash
git show e3-jsi/Monitoring-view:backend/app/services/websocket_manager.py > backend/app/services/websocket_manager.py
git show e3-jsi/Monitoring-view:backend/app/services/bioner_client.py > backend/app/services/bioner_client.py
```

- [ ] **Step 2: Reconcile imports/settings**

Open both files. Confirm every `from app...` import resolves against the current tree. In `bioner_client.py`, confirm the bioner base URL setting name matches `backend/app/core/settings.py` (compare against the Monitoring-view `settings.py` diff: `git show e3-jsi/Monitoring-view:backend/app/core/settings.py | grep -i bioner`). If the setting is missing, add it to `core/settings.py` mirroring the existing `BIONER_*` settings pattern. Remove any reference to dropped tables (these two files reference none — verify with `grep -nE "TrainingEvaluation|ModelArtifact|UserModelPreference" backend/app/services/websocket_manager.py backend/app/services/bioner_client.py`; expect no matches).

- [ ] **Step 3: Verify import**

Run: `cd backend && python -c "import app.services.websocket_manager, app.services.bioner_client"`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/websocket_manager.py backend/app/services/bioner_client.py backend/app/core/settings.py
git commit -m "feat(training): port websocket manager and bioner trainer client"
```

---

## Phase 4 — Backend routes + schemas

### Task 7: Training/monitoring schemas

**Files:**
- Modify: `backend/app/schemas.py`

- [ ] **Step 1: Add request/response schemas**

Append to `backend/app/schemas.py`:
```python
class GLiNERTrainingRequest(BaseModel):
    """Request body to start a GLiNER training run."""

    dataset_id: Optional[int] = None
    labels: List[str] = Field(default_factory=list)
    base_model: str = "urchade/gliner_small-v2.1"
    val_ratio: float = 0.1


class TrainingStartResponse(BaseModel):
    run_id: int


class TrainingRunSummary(BaseModel):
    run_id: int
    status: str


class RunEvaluationResponse(BaseModel):
    run_id: int
    per_label: Dict[str, Dict[str, Any]]


class FullStatsResponse(BaseModel):
    totalRecords: int
    totalTerms: int
    labelDistribution: Dict[str, int]
```
Ensure the file imports `Any, Dict, List, Optional` from `typing`.

- [ ] **Step 2: Verify import**

Run: `cd backend && python -c "import app.schemas"`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add backend/app/schemas.py
git commit -m "feat(training): add training/monitoring API schemas"
```

### Task 8: Training/monitoring routes under `/bioner`

**Files:**
- Modify: `backend/app/routes/v1/bioner.py`
- Test: `backend/tests/test_training_routes.py`

**Endpoints (paths the existing frontend calls):**
- `POST /bioner/training/start` → `{run_id}`
- `POST /bioner/training/stop/{run_id}` → `{message}`
- `GET /bioner/datasets/{dataset_id}/full-stats` → `{totalRecords, totalTerms, labelDistribution}`
- `GET /bioner/datasets/{dataset_id}/runs` → `[{run_id, status}]`
- `GET /bioner/runs/{run_id}/evaluation` → `{run_id, per_label}`
- `GET /bioner/datasets/{dataset_id}/runs/evaluations` → `[{run_id, per_label}]`
- `WS /bioner/ws/training`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_training_routes.py
def test_start_creates_pending_run(client, auth_headers, sample_dataset):
    resp = client.post("/api/v1/bioner/training/start", headers=auth_headers, json={
        "dataset_id": sample_dataset.id, "labels": ["Drug"],
        "base_model": "urchade/gliner_small-v2.1", "val_ratio": 0.1,
    })
    assert resp.status_code == 200
    assert "run_id" in resp.json()


def test_full_stats_shape(client, auth_headers, sample_dataset, sample_record):
    resp = client.get(
        f"/api/v1/bioner/datasets/{sample_dataset.id}/full-stats", headers=auth_headers
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"totalRecords", "totalTerms", "labelDistribution"}
    assert body["totalRecords"] >= 1


def test_run_evaluation_shape(client, auth_headers, session, sample_dataset, sample_record):
    from app.services import training_service as svc
    run = svc.create_run(session, dataset_id=sample_dataset.id, base_model="b",
                         labels=["Drug"], val_ratio=0.0)
    svc.record_evaluation(session, run.id, {"Drug": {"exact_f1": 0.8, "relaxed_f1": 0.9}})
    resp = client.get(f"/api/v1/bioner/runs/{run.id}/evaluation", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["per_label"]["Drug"]["exact_f1"] == 0.8
```
(Reuse the `client`/`auth_headers` fixtures from `test_datasets.py`/`test_login.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_training_routes.py -v`
Expected: FAIL — 404 on the new routes.

- [ ] **Step 3: Mock the trainer call so `start` doesn't hit the bioner service**

In the test file, monkeypatch `app.services.bioner_client.start_training` to a no-op returning `None` (add an `autouse` fixture in this module). Example:
```python
import pytest

@pytest.fixture(autouse=True)
def _no_trainer(monkeypatch):
    monkeypatch.setattr("app.services.bioner_client.start_training", lambda *a, **k: None)
```

- [ ] **Step 4: Add the route handlers to `bioner.py`**

Add (thin layer delegating to services). At top, extend imports:
```python
from fastapi import WebSocket, WebSocketDisconnect
from sqlmodel import func, select
from app.models_db import Record, SourceTerm, TrainingRun
from app.schemas import (
    FullStatsResponse, GLiNERTrainingRequest, RunEvaluationResponse,
    TrainingRunSummary, TrainingStartResponse,
)
from app.services import bioner_client, evaluation_service, training_service
from app.services.websocket_manager import manager  # name per ported module
```
Then the handlers:
```python
@router.post("/training/start", response_model=TrainingStartResponse)
def start_training(req: GLiNERTrainingRequest, db: Session = Depends(get_session)):
    run = training_service.create_run(
        db, dataset_id=req.dataset_id, base_model=req.base_model,
        labels=req.labels, val_ratio=req.val_ratio,
    )
    bioner_client.start_training(run_id=run.id, dataset_id=req.dataset_id,
                                 labels=req.labels, base_model=req.base_model,
                                 val_ratio=req.val_ratio)
    return TrainingStartResponse(run_id=run.id)


@router.post("/training/stop/{run_id}")
def stop_training(run_id: int, db: Session = Depends(get_session)):
    training_service.stop_run(db, run_id)
    return {"message": "stopped"}


@router.get("/datasets/{dataset_id}/full-stats", response_model=FullStatsResponse)
def full_stats(dataset_id: int, db: Session = Depends(get_session)):
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
        totalRecords=total_records, totalTerms=total_terms,
        labelDistribution={label: count for label, count in rows},
    )


@router.get("/datasets/{dataset_id}/runs", response_model=list[TrainingRunSummary])
def list_runs(dataset_id: int, db: Session = Depends(get_session)):
    runs = db.exec(
        select(TrainingRun).where(TrainingRun.dataset_id == dataset_id)
        .order_by(TrainingRun.id.desc())
    ).all()
    return [TrainingRunSummary(run_id=r.id, status=r.status) for r in runs]


@router.get("/runs/{run_id}/evaluation", response_model=RunEvaluationResponse)
def run_evaluation(run_id: int, db: Session = Depends(get_session)):
    run = db.get(TrainingRun, run_id)
    per_label = {}
    if run is not None and run.model_id is not None:
        per_label = evaluation_service.get_per_label(db, run.model_id)
    return RunEvaluationResponse(run_id=run_id, per_label=per_label)


@router.get("/datasets/{dataset_id}/runs/evaluations")
def dataset_runs_evaluations(dataset_id: int, db: Session = Depends(get_session)):
    runs = db.exec(
        select(TrainingRun).where(TrainingRun.dataset_id == dataset_id)
    ).all()
    out = []
    for r in runs:
        per_label = evaluation_service.get_per_label(db, r.model_id) if r.model_id else {}
        out.append({"run_id": r.id, "per_label": per_label})
    return out


@router.websocket("/ws/training")
async def ws_training(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```
Verify `router`, `get_session`, `Session`, `Depends` are already imported in `bioner.py` (they are, for extraction). Match the auth dependency the other `bioner.py` routes use (e.g. `Depends(get_current_user)`) on the HTTP endpoints.

- [ ] **Step 5: Wire trainer events into the lifecycle**

The trainer posts events to a callback endpoint (`training-events`) or via websocket. Port the event handler from `git show e3-jsi/Monitoring-view:backend/app/routes/v1/training_events.py`, then **rewrite its persistence** to call `training_service`:
- `training_info` → `training_service.mark_running(db, run_id)`
- `epoch_update` → `training_service.add_epoch_metric(db, run_id, epoch, loss)`
- `evaluation_completed` → `training_service.record_evaluation(db, run_id, payload["per_label"])`
- `completed` → `training_service.complete_run(db, run_id, payload["output_path"], record_ids)` where `record_ids` = the dataset's record ids used for training (query `Record.id` for the run's `dataset_id`)
- `error` → `training_service.fail_run(db, run_id, payload.get("message", "training failed"))`
- `stopped` → `training_service.stop_run(db, run_id)`
Delete all references to `TrainingEvaluation`/`ModelArtifact`. Broadcast each event via `manager.broadcast(payload)` for the websocket feed. Register this router/endpoint in `routes/v1/__init__.py` if it is a separate router, or add it to `bioner.py`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_training_routes.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Run the full backend suite**

Run: `cd backend && pytest -q`
Expected: all pass (existing + new).

- [ ] **Step 8: Lint**

Run: `cd backend && ruff check --fix . && ruff format .`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add backend/app/routes/v1/bioner.py backend/app/routes/v1/training_events.py backend/app/routes/v1/__init__.py backend/tests/test_training_routes.py
git commit -m "feat(training): training/monitoring routes under /bioner with tests"
```

### Task 9: Remove the dead `monitoring.py` router (if present in tree)

**Files:**
- Delete: `backend/app/routes/v1/monitoring.py` (only if it was carried in; ninakokalj base does not have it — verify first)

- [ ] **Step 1: Verify it is not registered/used**

Run: `grep -rn "monitoring" backend/app/routes/v1/__init__.py backend/app/main.py`
Expected: no monitoring router registration. If a `monitoring.py` file exists and is unregistered, delete it. If it does not exist (ninakokalj base), skip this task.

- [ ] **Step 2: Commit (if a deletion happened)**

```bash
git rm backend/app/routes/v1/monitoring.py
git commit -m "chore(training): drop unused monitoring router (consolidated into /bioner)"
```

---

## Phase 5 — bioner service (trainer)

### Task 10: Port the GLiNER trainer into the `bioner` service

**Files:**
- Create: `bioner/app/library/ner_metrics.py`
- Create: `bioner/app/training/__init__.py`, `trainer.py`, `gliner_trainer.py`, `job_manager.py`, `callbacks.py`
- Create/Modify: `bioner/app/routes/v1/training.py` and/or `bioner/app/main.py` (match base layout)

- [ ] **Step 1: Port files verbatim**

```bash
for f in library/ner_metrics.py training/__init__.py training/trainer.py \
         training/gliner_trainer.py training/job_manager.py training/callbacks.py; do
  mkdir -p "bioner/app/$(dirname "$f")"
  git show "e3-jsi/Monitoring-view:bioner/app/$f" > "bioner/app/$f"
done
```

- [ ] **Step 2: Reconcile the trainer's emitted events to the lifecycle**

Open `bioner/app/training/gliner_trainer.py` and `callbacks.py`. Confirm the events it posts back match the backend handler (Task 8 Step 5): `training_info`, `epoch_update {epoch, loss}`, `evaluation_completed {per_label}`, `completed {output_path}`, `error {message}`, `stopped`. The `evaluation_completed.per_label` must be `{label: {exact_f1, relaxed_f1, precision, recall}}` — if the trainer computes only one F1, use `ner_metrics.NERMetrics(["exact", "relaxed"])` to produce both and shape the dict accordingly. Adjust field names to match; do not change training logic.

- [ ] **Step 3: Reconcile training route registration**

Compare `git show e3-jsi/Monitoring-view:bioner/app/main.py` against the current `bioner/app/main.py`. Port only the training endpoint registration + request handling (start/stop), matching the current file's router/registration style. Do not port model-management routes (`routes_model_management.py`, `model_manager.py`) — deferred bucket C.

- [ ] **Step 4: Install/lock deps**

Compare `git show e3-jsi/Monitoring-view:bioner/requirements.txt` with the current one; add only the training deps actually imported by the ported files (e.g. `gliner`, training extras). Run: `cd bioner && pip install -r requirements.txt`. Expected: success.

- [ ] **Step 5: Smoke-import**

Run: `cd bioner && python -c "import app.training.gliner_trainer, app.training.job_manager"`
Expected: exit 0 (heavy ML imports may be slow; failures = missing deps to add in Step 4).

- [ ] **Step 6: Commit**

```bash
git add bioner/app/library/ner_metrics.py bioner/app/training bioner/app/main.py bioner/requirements.txt
git commit -m "feat(bioner): port GLiNER trainer, job manager, and training routes"
```

---

## Phase 6 — End-to-end verification

### Task 11: Integration smoke through docker-compose

**Files:** none (verification only)

- [ ] **Step 1: Bring up the stack**

Run: `docker-compose up -d --build`
Expected: postgres, elasticsearch, backend, bioner, frontend healthy.

- [ ] **Step 2: Apply migrations**

Run: `docker-compose exec backend alembic upgrade head`
Expected: head at rev `010`.

- [ ] **Step 3: Drive a training run via the API**

With a logged-in token and a dataset that has records+source terms, `POST /api/v1/bioner/training/start`, then poll `GET /api/v1/bioner/datasets/{id}/runs` until status leaves `pending`, and confirm `GET /api/v1/bioner/runs/{run_id}/evaluation` returns a `per_label` dict containing `exact_f1` and `relaxed_f1` once evaluation completes. Capture the run reaching `completed` and a `Model` row with a `path`.

- [ ] **Step 4: Confirm DB rows**

Via Adminer (localhost:8080) or psql, confirm rows in `training_run` (status `completed`), `training_metric` (loss per epoch), `model` (path set), `evaluation` (one row per label), `model_train_record_link`.

- [ ] **Step 5: Commit any fixups**

```bash
git add -A && git commit -m "fix(training): integration fixups from e2e run"
```

---

## Phase 7 — Frontend wiring (minimal; full UI is deferred)

### Task 12: Point the Monitor API client at `/bioner` and keep it functional

**Files:**
- Modify: `frontend/src/api/monitoring.ts`
- Bring in (port): `frontend/src/pages/Monitor/index.tsx`, `frontend/src/pages/Monitor/LabelSelector.tsx`, monitoring types in `frontend/src/types/index.ts`, the Monitor route in `frontend/src/pages/App/index.tsx` and the Sidebar link.

- [ ] **Step 1: Port the Monitor page + api + types**

```bash
for f in pages/Monitor/index.tsx pages/Monitor/LabelSelector.tsx api/monitoring.ts; do
  mkdir -p "frontend/src/$(dirname "$f")"
  git show "e3-jsi/Monitoring-view:frontend/src/$f" > "frontend/src/$f"
done
```
Add the monitoring-related interfaces from `git show e3-jsi/Monitoring-view:frontend/src/types/index.ts` into the current `types/index.ts` (only the training/monitoring ones).

- [ ] **Step 2: Confirm API paths + response shapes**

In `api/monitoring.ts`, ensure all calls hit `/bioner/*` paths matching Task 8. The evaluation `per_label` shape now includes `exact_f1`/`relaxed_f1`; keep the existing `precision`/`recall` reads working (they remain present). Leave deeper UI rendering of exact/relaxed F1 to the deferred UI pass.

- [ ] **Step 3: Add the route + sidebar entry**

Port the Monitor lazy route into `frontend/src/pages/App/index.tsx` and the Sidebar nav item from `git show e3-jsi/Monitoring-view:frontend/src/components/Sidebar/index.tsx` (only the Monitor entry).

- [ ] **Step 4: Typecheck + build + lint**

Run: `cd frontend && npm run build && npm run lint`
Expected: build succeeds, lint clean.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/monitoring.ts frontend/src/pages/Monitor frontend/src/types/index.ts frontend/src/pages/App/index.tsx frontend/src/components/Sidebar/index.tsx
git commit -m "feat(monitor): wire Monitor page to /bioner endpoints (UI polish deferred)"
```

---

## Self-Review notes

- **Spec coverage:** schema (T1), migration (T2), exact/relaxed F1 (T3), Evaluation JSON (T4), lifecycle producing Model + ModelTrainRecordLink (T5), live feed infra (T6), schemas (T7), `/bioner` routes incl. full-stats/runs/evaluation/ws (T8), drop dead router (T9), trainer port (T10), e2e (T11), minimal frontend (T12). Deferred items (UserModelPreference, ModelArtifact, TrainingEvaluation, SourceTermLink, model-settings, storybook) explicitly excluded.
- **Type consistency:** service signatures in T5 match calls in T8 Step 5; `per_label` shape consistent across T4/T5/T8/T10; `Model` fields from T1 used in T5/T8.
- **Known adaptation risks (call out during execution):** exact bioner trainer event field names (T10 Step 2), settings name for the bioner URL (T6 Step 2), existing conftest fixture names (T5 Step 1). These require reading the actual files at execution time and adjusting.
