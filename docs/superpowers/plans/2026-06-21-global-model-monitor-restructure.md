# Global Model + Monitor Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the NER extraction model a single instance-global selection chosen in Monitor (not per-dataset in Term Extraction), restructure Monitor's Comparison view into a global model list + per-model detail, and upgrade the Training tab with curated GLiNER baselines and a step-indexed train+eval loss curve.

**Architecture:** A new single-row `app_settings` table holds the global `active_model_id`; the per-dataset `Dataset.active_model_id` column is dropped. Extraction resolves the one global model (pinned per job; switching is blocked while any extraction job runs). Bioner keeps its single loaded model and only hot-swaps when the global selection changes. The trainer gains a real `eval_dataset` + `eval_strategy="steps"` so an eval-loss curve streams alongside train loss; training-time dataset stats are snapshotted onto the run.

**Tech Stack:** Backend FastAPI + SQLModel + Alembic + Postgres. Bioner GLiNER (HF Trainer) + LitServe. Frontend React 19 + Vite + TypeScript (strict) + existing chart wrappers in `frontend/src/components/charts/`.

## Global Constraints

- Python services: Python 3.10+, `ruff` for lint/format, `pytest`. Config per-service `pyproject.toml`.
- Frontend: TypeScript strict, Prettier (double quotes, 2-space, `printWidth: 120`, `trailingComma: es5`), CSS Modules, package manager **npm**. Path aliases: `@/`, `@components/`, `@pages/`, `@hooks/`, `@api/`, `@types/`, `@assets/`.
- **Do not hardcode service hosts** — read from settings/env (`EXTRACT_HOST`, `BACKEND_HOST`).
- Postgres is authoritative; Elasticsearch untouched here.
- **Local exec constraint:** the backend Python venv (3.10) is gone on this host — run backend tests/migrations via Docker: `docker compose exec backend pytest ...` and `docker compose exec backend alembic ...`. The `ruff` binary still works locally for lint. Verify frontend changes against the Docker frontend (rebuild/restart) or a Vite dev server pointed at the worktree.
- **Alembic head is `006`** (`backend/alembic/versions/006_dataset_active_model.py`). New migrations chain from there.
- **Curated GLiNER baselines (default first):**
  - `urchade/gliner_multi-v2.1` — Multilingual (DEFAULT)
  - `urchade/gliner_large-v2.1` — Large (best/slow, English-centric)
  - `E3-JSI/gliner-multi-med-ner-synthetic-v1` — Biomedical/clinical multilingual (already the stack's runtime default; GLiNER-loadable, no external risk)
  - Custom path — behind an "⚠ Advanced" warning
- **Two distinct defaults coexist by design:** the bioner *runtime* default (`BIONER_MODEL`, currently `E3-JSI/gliner-multi-med-ner-synthetic-v1`) vs. the *training-baseline* default (`urchade/gliner_multi-v2.1`). Not a conflict.

---

## Decisions locked during design (authoritative)

1. Global active model stored in new `app_settings` single row; `Dataset.active_model_id` **dropped**.
2. Extraction **pins** the model at job start; **block** changing the global model while *any* extraction job is active (instance-wide).
3. Truly global, all users share one model. Bioner keeps one loaded model; swap only on global change (no per-request activation in the extraction hot path).
4. Monitor model list = **trained runs only** + a synthetic **"Default (bioner)"** entry. Per-dataset `"Base model"` baseline rows are **hidden from the list** (still stored, still used for the in-detail base-vs-trained comparison).
5. Per-model detail keeps **base-vs-trained** delta (same split). All **cross-model** widgets removed (leaderboard, multi-run loss overlay, cross-run eval bars, runs×labels heatmap).
6. Remove Monitor's page-level `DatasetSelector`. Models tab is global; Training tab keeps its own dataset multi-select.
7. Training-time dataset stats + label coverage are **snapshotted** into new `TrainingRun.train_stats` JSON.
8. Term Extraction shows the active model as a **read-only tag/chip** with a link to Monitor.
9. Hyperparameters unchanged: epochs, learning rate, batch size, validation split.
10. Eval curve via `eval_strategy="steps"`; N tuned for ~15–20 eval points. Persist step-indexed train+eval metrics so the plot survives reload.
11. Run-list affordances: keep **Use-for-extraction** (now the global setter), **Rename**, **Delete**. Drop the **preferred-star** and the **error-analysis** panel.

---

# Phase 1 — Backend data layer

## Task 1: `app_settings` table + global active model column

**Files:**
- Modify: `backend/app/models_db.py` (add `AppSettings` model; remove `Dataset.active_model_id`)
- Create: `backend/alembic/versions/007_global_active_model.py`
- Test: `backend/app/tests/test_app_settings_model.py`

**Interfaces:**
- Produces: `AppSettings(id=1, active_model_id: Optional[int])` SQLModel table `app_settings`; helper `get_app_settings(db) -> AppSettings` and `set_global_active_model(db, model_id) -> AppSettings` (Task 5 consumes these).

- [ ] **Step 1: Write the failing test**

```python
# backend/app/tests/test_app_settings_model.py
from sqlmodel import Session, SQLModel, create_engine
from app.models_db import AppSettings


def test_app_settings_singleton_defaults_to_null_model():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        row = AppSettings(id=1, active_model_id=None)
        db.add(row)
        db.commit()
        db.refresh(row)
        assert row.id == 1
        assert row.active_model_id is None
```

- [ ] **Step 2: Run test, verify it fails**

Run: `docker compose exec backend pytest app/tests/test_app_settings_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'AppSettings'`.

- [ ] **Step 3: Add the `AppSettings` model and drop the per-dataset column**

In `backend/app/models_db.py`, add near the other table models:

```python
class AppSettings(SQLModel, table=True):
    """Single-row, instance-wide application settings.

    Always uses ``id == 1``. Holds the GLOBAL active NER extraction model shared
    by all users (null = bioner's launch default model).
    """

    __tablename__ = "app_settings"

    id: Optional[int] = Field(default=1, primary_key=True)
    active_model_id: Optional[int] = Field(
        default=None, foreign_key="model.id", ondelete="SET NULL", nullable=True
    )
```

In the `Dataset` class, **delete** these lines:

```python
    # Trained model selected for NER extraction on this dataset (null = bioner default).
    active_model_id: Optional[int] = Field(
        default=None, foreign_key="model.id", ondelete="SET NULL", nullable=True
    )
```

- [ ] **Step 4: Run test, verify it passes**

Run: `docker compose exec backend pytest app/tests/test_app_settings_model.py -v`
Expected: PASS.

- [ ] **Step 5: Write the Alembic migration**

```python
# backend/alembic/versions/007_global_active_model.py
"""global active model: add app_settings, drop dataset.active_model_id

Revision ID: 007
Revises: 006
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "app_settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("active_model_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["active_model_id"], ["model.id"], ondelete="SET NULL"
        ),
    )
    # Seed the singleton row.
    op.execute("INSERT INTO app_settings (id, active_model_id) VALUES (1, NULL)")
    # Drop the per-dataset active model column (FK constraint first if named).
    with op.batch_alter_table("dataset") as batch:
        batch.drop_column("active_model_id")


def downgrade() -> None:
    op.add_column(
        "dataset",
        sa.Column("active_model_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_dataset_active_model_id",
        "dataset",
        "model",
        ["active_model_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.drop_table("app_settings")
```

> Note: if Postgres named the `dataset.active_model_id` FK constraint, drop it explicitly before `drop_column` (check `006_dataset_active_model.py` for the constraint name; mirror it). Inspect with `docker compose exec backend alembic upgrade 006` then `\d dataset` in Adminer if unsure.

- [ ] **Step 6: Apply + verify migration**

Run:
```bash
docker compose exec backend alembic upgrade head
docker compose exec backend alembic current
```
Expected: `current` reports `007 (head)`. In Adminer, `dataset` has no `active_model_id`, `app_settings` has one row `(1, NULL)`.

- [ ] **Step 7: Commit**

```bash
git add backend/app/models_db.py backend/alembic/versions/007_global_active_model.py backend/app/tests/test_app_settings_model.py
git commit -m "feat(backend): add app_settings global active model, drop per-dataset active_model_id"
```

---

## Task 2: `TrainingRun.train_stats` snapshot column

**Files:**
- Modify: `backend/app/models_db.py` (`TrainingRun` gains `train_stats`)
- Create: `backend/alembic/versions/008_training_run_train_stats.py`
- Test: `backend/app/tests/test_training_run_train_stats.py`

**Interfaces:**
- Produces: `TrainingRun.train_stats: Optional[dict]` (JSON). Shape (written by Task 6):
  ```json
  {"train_dataset_ids": [int], "eval_dataset_ids": [int],
   "record_count": int, "term_count": int,
   "label_distribution": {"<label>": int},
   "train_size": int, "eval_size": int, "val_ratio": float}
  ```

- [ ] **Step 1: Write the failing test**

```python
# backend/app/tests/test_training_run_train_stats.py
from sqlmodel import Session, SQLModel, create_engine
from app.models_db import TrainingRun, Dataset, User


def _setup(db):
    user = User(email="a@b.c", hashed_password="x")
    db.add(user); db.commit(); db.refresh(user)
    ds = Dataset(name="d", labels=[], user_id=user.id)
    db.add(ds); db.commit(); db.refresh(ds)
    return ds


def test_training_run_stores_train_stats_json():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        ds = _setup(db)
        run = TrainingRun(
            dataset_id=ds.id, base_model="urchade/gliner_multi-v2.1",
            labels=["Drug"], val_ratio=0.1,
            train_stats={"record_count": 12, "term_count": 40,
                         "label_distribution": {"Drug": 40}},
        )
        db.add(run); db.commit(); db.refresh(run)
        assert run.train_stats["record_count"] == 12
        assert run.train_stats["label_distribution"]["Drug"] == 40
```

- [ ] **Step 2: Run test, verify it fails**

Run: `docker compose exec backend pytest app/tests/test_training_run_train_stats.py -v`
Expected: FAIL (`TypeError: 'train_stats' is an invalid keyword argument`).

- [ ] **Step 3: Add the column**

In `backend/app/models_db.py`, in `TrainingRun`, after `created_at`:

```python
    # Snapshot of the training datasets' stats AT TRAINING TIME (datasets mutate
    # afterward). Shape: {train_dataset_ids, eval_dataset_ids, record_count,
    # term_count, label_distribution, train_size, eval_size, val_ratio}.
    train_stats: Optional[dict] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
```

(`Column` and `JSON` are already imported at the top of the file.)

- [ ] **Step 4: Run test, verify it passes**

Run: `docker compose exec backend pytest app/tests/test_training_run_train_stats.py -v`
Expected: PASS.

- [ ] **Step 5: Write the migration**

```python
# backend/alembic/versions/008_training_run_train_stats.py
"""training_run.train_stats snapshot column

Revision ID: 008
Revises: 007
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "training_run",
        sa.Column("train_stats", postgresql.JSON(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("training_run", "train_stats")
```

- [ ] **Step 6: Apply + verify**

Run: `docker compose exec backend alembic upgrade head && docker compose exec backend alembic current`
Expected: `008 (head)`; `training_run.train_stats` column exists.

- [ ] **Step 7: Commit**

```bash
git add backend/app/models_db.py backend/alembic/versions/008_training_run_train_stats.py backend/app/tests/test_training_run_train_stats.py
git commit -m "feat(backend): add TrainingRun.train_stats snapshot column"
```

---

## Task 3: Step-indexed training metrics (train + eval loss)

The current `TrainingMetric` is per-epoch `(epoch, loss)` only. The eval curve needs step-indexed points carrying an optional `eval_loss`. Extend the table additively (keep `epoch`, `loss`).

**Files:**
- Modify: `backend/app/models_db.py` (`TrainingMetric` gains `step`, `eval_loss`)
- Create: `backend/alembic/versions/009_training_metric_step_eval.py`
- Modify: `backend/app/services/training_service.py` (`add_epoch_metric` → add `add_step_metric`)
- Test: `backend/app/tests/test_training_metric_step.py`

**Interfaces:**
- Produces: `TrainingMetric(run_id, epoch, loss, step: Optional[int], eval_loss: Optional[float])` and
  `training_service.add_step_metric(db, run_id, *, step, epoch, loss=None, eval_loss=None) -> None`
  (Task 8 consumes `add_step_metric`).

- [ ] **Step 1: Write the failing test**

```python
# backend/app/tests/test_training_metric_step.py
from sqlmodel import Session, SQLModel, create_engine, select
from app.models_db import TrainingMetric, TrainingRun, Dataset, User
from app.services import training_service


def _run(db):
    u = User(email="a@b.c", hashed_password="x"); db.add(u); db.commit(); db.refresh(u)
    ds = Dataset(name="d", labels=[], user_id=u.id); db.add(ds); db.commit(); db.refresh(ds)
    r = TrainingRun(dataset_id=ds.id, base_model="m", labels=["Drug"], val_ratio=0.1)
    db.add(r); db.commit(); db.refresh(r)
    return r


def test_add_step_metric_persists_step_and_eval_loss():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        r = _run(db)
        training_service.add_step_metric(db, r.id, step=10, epoch=1, loss=0.5)
        training_service.add_step_metric(db, r.id, step=20, epoch=1, eval_loss=0.42)
        rows = db.exec(select(TrainingMetric).where(TrainingMetric.run_id == r.id)
                       .order_by(TrainingMetric.step)).all()
        assert [m.step for m in rows] == [10, 20]
        assert rows[0].loss == 0.5 and rows[0].eval_loss is None
        assert rows[1].eval_loss == 0.42 and rows[1].loss is None
```

- [ ] **Step 2: Run test, verify it fails**

Run: `docker compose exec backend pytest app/tests/test_training_metric_step.py -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'add_step_metric'`).

- [ ] **Step 3: Extend `TrainingMetric`**

In `backend/app/models_db.py`, in `TrainingMetric`, after `loss`:

```python
    # Step-indexed metrics for the live train/eval loss curve. ``step`` is the
    # trainer global step; ``eval_loss`` is populated only on eval-step rows.
    step: Optional[int] = Field(default=None, index=True)
    eval_loss: Optional[float] = Field(default=None)
```

- [ ] **Step 4: Add `add_step_metric` in `training_service.py`**

Below the existing `add_epoch_metric`:

```python
def add_step_metric(
    db: Session,
    run_id: int,
    *,
    step: int,
    epoch: int,
    loss: Optional[float] = None,
    eval_loss: Optional[float] = None,
) -> None:
    """Append a step-indexed metric point (train loss and/or eval loss).

    Train-step rows carry ``loss``; eval-step rows carry ``eval_loss``. Either may
    be None; the row is still written so the curve keeps step alignment.
    """
    db.add(
        TrainingMetric(
            run_id=run_id, epoch=epoch, step=step, loss=loss, eval_loss=eval_loss
        )
    )
    db.commit()
```

- [ ] **Step 5: Run test, verify it passes**

Run: `docker compose exec backend pytest app/tests/test_training_metric_step.py -v`
Expected: PASS.

- [ ] **Step 6: Write the migration**

```python
# backend/alembic/versions/009_training_metric_step_eval.py
"""training_metric: add step + eval_loss

Revision ID: 009
Revises: 008
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("training_metric", sa.Column("step", sa.Integer(), nullable=True))
    op.add_column("training_metric", sa.Column("eval_loss", sa.Float(), nullable=True))
    op.create_index("ix_training_metric_step", "training_metric", ["step"])


def downgrade() -> None:
    op.drop_index("ix_training_metric_step", table_name="training_metric")
    op.drop_column("training_metric", "eval_loss")
    op.drop_column("training_metric", "step")
```

- [ ] **Step 7: Apply + verify**

Run: `docker compose exec backend alembic upgrade head && docker compose exec backend alembic current`
Expected: `009 (head)`.

- [ ] **Step 8: Commit**

```bash
git add backend/app/models_db.py backend/alembic/versions/009_training_metric_step_eval.py backend/app/services/training_service.py backend/app/tests/test_training_metric_step.py
git commit -m "feat(backend): step-indexed training metrics with eval_loss"
```

---

# Phase 2 — Backend API

## Task 4: Global active-model service helpers + active-extraction guard

**Files:**
- Modify: `backend/app/services/training_service.py` (add `get_app_settings`, `set_global_active_model`, `get_global_active_model`)
- Create: `backend/app/services/extraction_lock.py` (instance-wide active-extraction check)
- Test: `backend/app/tests/test_global_active_model_service.py`

**Interfaces:**
- Produces:
  - `training_service.get_global_active_model(db) -> Optional[Model]`
  - `training_service.set_global_active_model(db, model_id: Optional[int]) -> AppSettings`
  - `extraction_lock.any_extraction_job_active(db) -> bool` (True if any `ExtractionJob` is running/pending instance-wide)

- [ ] **Step 1: Inspect `ExtractionJob` status values**

Run: `docker compose exec backend python -c "from app.models_db import ExtractionJob; print(ExtractionJob.__fields__.keys())"` (or grep `class ExtractionJob` in `models_db.py`). Confirm the column that marks active state and its non-terminal values (e.g. `status in ("pending","running")`). Use the real values in Step 3.

- [ ] **Step 2: Write the failing test**

```python
# backend/app/tests/test_global_active_model_service.py
from sqlmodel import Session, SQLModel, create_engine
from app.models_db import AppSettings, Model, Dataset, User
from app.services import training_service


def test_set_and_get_global_active_model():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(AppSettings(id=1, active_model_id=None)); db.commit()
        u = User(email="a@b.c", hashed_password="x"); db.add(u); db.commit(); db.refresh(u)
        ds = Dataset(name="d", labels=[], user_id=u.id); db.add(ds); db.commit(); db.refresh(ds)
        m = Model(name="run-1", version="v", base_model="b", path="/models/run-1", dataset_id=ds.id)
        db.add(m); db.commit(); db.refresh(m)

        assert training_service.get_global_active_model(db) is None
        training_service.set_global_active_model(db, m.id)
        assert training_service.get_global_active_model(db).id == m.id
        training_service.set_global_active_model(db, None)
        assert training_service.get_global_active_model(db) is None
```

- [ ] **Step 3: Implement helpers**

In `training_service.py`:

```python
def get_app_settings(db: Session) -> "AppSettings":
    """Return the singleton AppSettings row, creating it if missing."""
    settings_row = db.get(AppSettings, 1)
    if settings_row is None:
        settings_row = AppSettings(id=1, active_model_id=None)
        db.add(settings_row)
        db.commit()
        db.refresh(settings_row)
    return settings_row


def get_global_active_model(db: Session) -> Optional[Model]:
    """Return the globally selected extraction Model, or None (= bioner default)."""
    row = get_app_settings(db)
    if row.active_model_id is None:
        return None
    return db.get(Model, row.active_model_id)


def set_global_active_model(db: Session, model_id: Optional[int]) -> "AppSettings":
    """Set or clear the global active extraction model."""
    row = get_app_settings(db)
    row.active_model_id = model_id
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
```

Add `AppSettings` to the `from app.models_db import (...)` block at the top of `training_service.py`.

Create `backend/app/services/extraction_lock.py`:

```python
"""Instance-wide guard for blocking model switches during active extraction."""
from sqlmodel import Session, select

from app.models_db import ExtractionJob

# Non-terminal extraction job states (adjust to the real enum values found in Step 1).
ACTIVE_EXTRACTION_STATES = ("pending", "running")


def any_extraction_job_active(db: Session) -> bool:
    """True if any extraction job is currently active anywhere in the instance."""
    rows = db.exec(
        select(ExtractionJob.id).where(
            ExtractionJob.status.in_(ACTIVE_EXTRACTION_STATES)
        )
    ).first()
    return rows is not None
```

- [ ] **Step 4: Run test, verify it passes**

Run: `docker compose exec backend pytest app/tests/test_global_active_model_service.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/training_service.py backend/app/services/extraction_lock.py backend/app/tests/test_global_active_model_service.py
git commit -m "feat(backend): global active-model service helpers + extraction-active guard"
```

---

## Task 5: Global active-model endpoints; remove per-dataset endpoints

Replace the per-dataset `GET/POST /bioner/datasets/{id}/active-model` with global `GET/POST /bioner/active-model`. The POST also performs the bioner hot-swap and is blocked while extraction is active.

**Files:**
- Modify: `backend/app/routes/v1/bioner.py` (remove 2 per-dataset handlers; add 2 global handlers; refactor `resolve_active_model`)
- Modify: `backend/app/schemas.py` (`ActiveModelResponse` loses `dataset_id`; keep `SetActiveModelRequest`)
- Test: `backend/app/tests/test_active_model_endpoints.py`

**Interfaces:**
- Produces:
  - `GET /bioner/active-model` → `ActiveModelResponse{active_model: Optional[ModelSummary]}`
  - `POST /bioner/active-model` body `SetActiveModelRequest{model_id}` → `ActiveModelResponse`; 409 if extraction active; activates on bioner.
  - `resolve_active_model(db) -> int` (no longer takes `dataset`) — used by the extract routes (Task 7).

- [ ] **Step 1: Update `ActiveModelResponse`**

In `backend/app/schemas.py`:

```python
class ActiveModelResponse(BaseModel):
    """The globally selected extraction model (null = bioner default)."""

    active_model: Optional[ModelSummary] = None
```

(Delete the `dataset_id` field. Keep `SetActiveModelRequest` unchanged.)

- [ ] **Step 2: Write the failing test**

```python
# backend/app/tests/test_active_model_endpoints.py
# Uses the project's existing FastAPI TestClient + auth fixtures (mirror an
# existing test in app/tests that hits /bioner/* with an authed client).
def test_get_active_model_defaults_to_none(authed_client):
    resp = authed_client.get("/api/v1/bioner/active-model")
    assert resp.status_code == 200
    assert resp.json()["active_model"] is None


def test_set_active_model_blocked_during_extraction(authed_client, monkeypatch):
    import app.routes.v1.bioner as bioner_routes
    monkeypatch.setattr(
        bioner_routes.extraction_lock, "any_extraction_job_active", lambda db: True
    )
    resp = authed_client.post("/api/v1/bioner/active-model", json={"model_id": None})
    assert resp.status_code == 409
```

> Find the real authed-client fixture name in `backend/app/tests/conftest.py` and adapt. If none exists, mirror the auth setup used by an existing `test_*bioner*` test.

- [ ] **Step 3: Add the global handlers; remove the per-dataset ones**

In `backend/app/routes/v1/bioner.py`, **delete** `get_dataset_active_model` and `set_dataset_active_model` (the two `/datasets/{dataset_id}/active-model` handlers). Add `import app.services.extraction_lock as extraction_lock` at the top, plus a `bioner_activate` helper and the handlers:

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `docker compose exec backend pytest app/tests/test_active_model_endpoints.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/routes/v1/bioner.py backend/app/schemas.py backend/app/tests/test_active_model_endpoints.py
git commit -m "feat(backend): global active-model endpoints, block switch during extraction"
```

---

## Task 6: Extraction uses the global model (pinned per job); snapshot train_stats

**Files:**
- Modify: `backend/app/routes/v1/bioner.py` (`resolve_active_model`; both extract routes; `start_training` snapshot)
- Modify: `backend/app/services/training_service.py` (`create_run` accepts + stores `train_stats`)
- Test: `backend/app/tests/test_resolve_active_model_global.py`

**Interfaces:**
- Consumes: `training_service.get_global_active_model` (Task 4).
- Produces: `resolve_active_model(db) -> int`; `create_run(..., train_stats: Optional[dict] = None)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/tests/test_resolve_active_model_global.py
from unittest.mock import patch, MagicMock
from sqlmodel import Session, SQLModel, create_engine
from app.models_db import AppSettings, Model, Dataset, User
from app.routes.v1.bioner import resolve_active_model


def test_resolve_active_model_uses_global_selection():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        u = User(email="a@b.c", hashed_password="x"); db.add(u); db.commit(); db.refresh(u)
        ds = Dataset(name="d", labels=[], user_id=u.id); db.add(ds); db.commit(); db.refresh(ds)
        m = Model(name="run-1", version="v", base_model="b", path="/models/run-1", dataset_id=ds.id)
        db.add(m); db.commit(); db.refresh(m)
        db.add(AppSettings(id=1, active_model_id=m.id)); db.commit()
        with patch("app.routes.v1.bioner.requests") as req:
            req.post.return_value = MagicMock(raise_for_status=lambda: None)
            assert resolve_active_model(db) == m.id
            req.post.assert_called()  # bioner activate was called
```

- [ ] **Step 2: Run test, verify it fails**

Run: `docker compose exec backend pytest app/tests/test_resolve_active_model_global.py -v`
Expected: FAIL (`resolve_active_model` still takes `(dataset, db)`).

- [ ] **Step 3: Refactor `resolve_active_model` to global**

Replace the body of `resolve_active_model` in `bioner.py`:

```python
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
```

- [ ] **Step 4: Update both extract routes' call sites**

In `bioner.py`, the single-record extract (`POST /{dataset_id}/records/{record_id}/extract`) and batch extract (`POST /{dataset_id}/records/extract`) currently call `resolve_active_model(dataset, db)`. Change both to `resolve_active_model(db)`. (The batch route already "resolves the active model once" and pins `model_id` through the job — that pinning behavior is exactly the per-job snapshot we want; just the call signature changes.)

- [ ] **Step 5: Snapshot `train_stats` in `start_training`**

In `start_training`, after the ownership checks and before `create_run`, build the snapshot using the existing stats service. The aggregated multi-dataset stats are produced by the same code path behind `POST /bioner/datasets/full-stats`; locate that service function (grep `full-stats` / `get_multi_dataset_stats` in `backend/app/services/`) and call it directly:

```python
    stats = monitor_stats_service.get_multi_dataset_stats(db, train_ids)  # adapt name
    train_stats_snapshot = {
        "train_dataset_ids": train_ids,
        "eval_dataset_ids": eval_ids,
        "record_count": stats.record_count,
        "term_count": stats.term_count,
        "label_distribution": stats.label_distribution,
        "val_ratio": req.val_ratio,
    }
```

Then pass it into `create_run(..., train_stats=train_stats_snapshot)`.

In `training_service.create_run`, add the parameter and set it on the run:

```python
def create_run(
    db: Session,
    *,
    dataset_ids: List[int],
    base_model: str,
    labels: List[str],
    val_ratio: float,
    eval_dataset_ids: Optional[List[int]] = None,
    train_stats: Optional[dict] = None,
) -> TrainingRun:
    ...
    run = TrainingRun(
        dataset_id=dataset_ids[0],
        base_model=base_model,
        labels=labels,
        val_ratio=val_ratio,
        status="pending",
        train_stats=train_stats,
    )
```

> `train_size`/`eval_size` are known only after the trainer's split; the trainer already emits them in `evaluation_completed`-adjacent logging. Leave those two keys out of the snapshot for v1 (the record/term/label distribution snapshot is the load-bearing part); revisit if needed.

- [ ] **Step 6: Run the full bioner-routes test module**

Run: `docker compose exec backend pytest app/tests/test_resolve_active_model_global.py app/tests/test_active_model_endpoints.py -v`
Expected: PASS. Also run any existing extraction route tests: `docker compose exec backend pytest app/tests/ -k extract -v` and fix call-site fallout.

- [ ] **Step 7: Commit**

```bash
git add backend/app/routes/v1/bioner.py backend/app/services/training_service.py backend/app/tests/test_resolve_active_model_global.py
git commit -m "feat(backend): extraction uses global model; snapshot train_stats on run"
```

---

## Task 7: Global model list + per-model detail payload

The list must be global (all trained models instance-wide) and hide the per-dataset `"Base model"` baseline rows. Add a per-model detail endpoint returning the run's datasets, `train_stats`, label coverage, and base-vs-trained eval, plus an `is_active` flag for the global selection.

**Files:**
- Modify: `backend/app/routes/v1/bioner.py` (`list_models` global + exclude baselines + `is_active`; add `GET /bioner/models/{model_id}/detail`)
- Modify: `backend/app/schemas.py` (`ModelSummary` gains `run_id`, `is_active`; add `ModelDetailResponse`)
- Test: `backend/app/tests/test_models_list_and_detail.py`

**Interfaces:**
- Produces:
  - `GET /bioner/models` → `ModelsOutput` of trained models only (no `"Base model"` rows), each with `run_id`, `is_active`.
  - `GET /bioner/models/{model_id}/detail` → `ModelDetailResponse{run_id, base_model, train_dataset_ids, eval_dataset_ids, train_stats, labels, per_label_trained, per_label_baseline}`.

- [ ] **Step 1: Extend `ModelSummary` + add `ModelDetailResponse`**

In `schemas.py`:

```python
class ModelSummary(BaseModel):
    """A trained NER model that can be selected for extraction."""

    id: int
    name: str
    version: str
    base_model: Optional[str] = None
    path: Optional[str] = None
    dataset_id: Optional[int] = None
    created_at: Optional[datetime] = None
    score: Optional[float] = None
    run_id: Optional[int] = None       # NEW: links a model to its training run
    is_active: bool = False            # NEW: is this the global active model?


class ModelDetailResponse(BaseModel):
    """Detail for one trained model (per-model view; no cross-model comparison)."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: int
    run_id: Optional[int] = None
    base_model: Optional[str] = None
    train_dataset_ids: List[int] = []
    eval_dataset_ids: List[int] = []
    train_stats: Optional[dict] = None
    labels: List[str] = []
    per_label_trained: Dict[str, Dict[str, Any]] = {}
    per_label_baseline: Dict[str, Dict[str, Any]] = {}
```

- [ ] **Step 2: Write the failing test**

```python
# backend/app/tests/test_models_list_and_detail.py
def test_models_list_excludes_baseline_rows_and_is_global(authed_client, seed_two_users_with_models):
    resp = authed_client.get("/api/v1/bioner/models")
    assert resp.status_code == 200
    names = [m["name"] for m in resp.json()["models"]]
    assert "Base model" not in names           # baselines hidden
    assert any(n.startswith("run-") for n in names)


def test_model_detail_returns_train_stats_and_base_vs_trained(authed_client, seed_model_with_eval):
    model_id = seed_model_with_eval
    resp = authed_client.get(f"/api/v1/bioner/models/{model_id}/detail")
    assert resp.status_code == 200
    body = resp.json()
    assert "train_stats" in body
    assert "per_label_trained" in body and "per_label_baseline" in body
```

> Adapt fixtures to the project's test conventions (mirror existing `test_*bioner*` seeding).

- [ ] **Step 3: Make `list_models` global, exclude baselines, set flags**

Replace `list_models` in `bioner.py`:

```python
@router.get("/models", response_model=ModelsOutput)
def list_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """List all trained models (instance-global, since the active model is global).

    Excludes per-dataset 'Base model' baseline rows (they are comparison anchors,
    not selectable models). Each summary carries its run id and whether it is the
    global active model.
    """
    active_model_id = training_service.get_app_settings(db).active_model_id
    models = db.exec(
        select(Model)
        .where(Model.path.is_not(None))
        .where(Model.name != training_service.BASELINE_MODEL_NAME)
        .order_by(Model.created_at.desc())
    ).all()
    summaries = []
    for m in models:
        summary = _model_summary(db, m)
        summary.run_id = m.training_run.id if m.training_run else None
        summary.is_active = m.id == active_model_id
        summaries.append(summary)
    return ModelsOutput(models=summaries)
```

> The previous join filtered by `current_user`. With a global selection the list is intentionally instance-wide; any user can select any trained model. If later you want per-owner deletion guards, enforce them in the delete handler, not the list.

- [ ] **Step 4: Add the detail endpoint**

```python
@router.get("/models/{model_id}/detail", response_model=ModelDetailResponse)
def model_detail(
    model_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Per-model detail: training datasets, snapshot stats, labels, and the
    base-vs-trained per-label evaluation (same split, the only valid comparison)."""
    model = db.get(Model, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    run = model.training_run
    per_label_trained = _scores_only(evaluation_service.get_per_label(db, model.id))
    per_label_baseline = {}
    train_ids: list[int] = []
    eval_ids: list[int] = []
    train_stats = None
    labels: list[str] = []
    if run is not None:
        train_ids = training_service.get_dataset_ids(db, run.id, role="train") or [run.dataset_id]
        eval_ids = training_service.get_dataset_ids(db, run.id, role="eval") or []
        train_stats = run.train_stats
        labels = run.labels or []
        baseline = training_service.get_baseline_model(db, run.dataset_id)
        if baseline is not None:
            per_label_baseline = _scores_only(evaluation_service.get_per_label(db, baseline.id))
    return ModelDetailResponse(
        model_id=model.id,
        run_id=run.id if run else None,
        base_model=model.base_model,
        train_dataset_ids=train_ids,
        eval_dataset_ids=eval_ids,
        train_stats=train_stats,
        labels=labels,
        per_label_trained=per_label_trained,
        per_label_baseline=per_label_baseline,
    )
```

> Confirm `training_service.get_dataset_ids(db, run_id, role=...)` exists (it is referenced in `training_events.py`). If not present, add it: query `TrainingRunDatasetLink` by `training_run_id` and `role`, return `dataset_id` list.

- [ ] **Step 5: Run tests, verify they pass**

Run: `docker compose exec backend pytest app/tests/test_models_list_and_detail.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/routes/v1/bioner.py backend/app/schemas.py backend/app/tests/test_models_list_and_detail.py
git commit -m "feat(backend): global model list (no baselines) + per-model detail endpoint"
```

---

# Phase 3 — Bioner trainer (eval curve)

## Task 8: Real eval loop + step-indexed eval-loss streaming + total steps

Wire the held-out/eval split as the Trainer's `eval_dataset`, enable `eval_strategy="steps"` with an auto-derived `eval_steps` (~15–20 eval points), and emit `total_steps` so the UI can show `step/total` + %.

**Files:**
- Modify: `bioner/app/training/gliner_trainer.py` (`_build_training_components`, `_build_trainer`, `_do_train`)
- Test: `bioner/app/tests/test_eval_curve_wiring.py`

**Interfaces:**
- Produces (events, consumed by Task 9):
  - `training_start` now also carries `total_steps: int`.
  - `train_log` rows carry `step`, `epoch`, optional `loss`, optional `eval_loss` (eval-step rows).

**Background (verified):** today no `eval_dataset` is passed to the Trainer and `TrainingArguments` sets no `eval_strategy`; the `eval_loss` key in `_build_trainer` is dead. The val split is a plain `list[dict]` from `_split_data`. We must build a GLiNER-compatible eval dataset and pass it in.

- [ ] **Step 1: Write the failing test**

```python
# bioner/app/tests/test_eval_curve_wiring.py
from app.training.gliner_trainer import GLiNERFinetuner


def test_compute_eval_steps_targets_about_15_to_20_points():
    f = GLiNERFinetuner(run_id=1, base_model_path="x", training_data=[{}])
    # 800 total steps -> ~16-20 eval points -> eval_steps in [40, 53]
    assert 40 <= f._compute_eval_steps(total_steps=800) <= 53
    # tiny runs still evaluate at least once
    assert f._compute_eval_steps(total_steps=3) >= 1


def test_total_steps_formula():
    f = GLiNERFinetuner(run_id=1, base_model_path="x", training_data=[{}],
                        num_epochs=4, train_batch_size=8)
    # 100 train items, batch 8 -> 13 steps/epoch * 4 = 52
    assert f._compute_total_steps(train_size=100) == 52
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cd bioner && python -m pytest app/tests/test_eval_curve_wiring.py -v` (bioner venv exists per its CLAUDE.md). If the venv isn't set up: `cd bioner && python -m venv .venv && . .venv/bin/activate && pip install -e .[test]`.
Expected: FAIL (`_compute_eval_steps`/`_compute_total_steps` undefined).

- [ ] **Step 3: Add the step-math helpers**

In `GLiNERFinetuner`:

```python
    import math  # ensure available at module top

    def _compute_total_steps(self, train_size: int) -> int:
        """Total optimizer steps = ceil(train_size / batch) * epochs (>=1)."""
        steps_per_epoch = max(1, math.ceil(train_size / max(1, self.train_batch_size)))
        return max(1, steps_per_epoch * self.num_epochs)

    def _compute_eval_steps(self, total_steps: int) -> int:
        """Eval interval targeting ~18 eval points across the run (>=1)."""
        TARGET_POINTS = 18
        return max(1, total_steps // TARGET_POINTS)
```

- [ ] **Step 4: Run helper tests, verify they pass**

Run: `cd bioner && python -m pytest app/tests/test_eval_curve_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Build a GLiNER eval dataset + enable eval in `TrainingArguments`**

In `_build_training_components`, after `train_ds = GLiNERDataset(train_data)` add an eval dataset built from `val_data` (only when non-empty), and parametrize `TrainingArguments` with eval settings:

```python
        eval_ds = GLiNERDataset(val_data) if val_data else None

        total_steps = self._compute_total_steps(len(train_data))
        eval_steps = self._compute_eval_steps(total_steps)

        eval_kwargs = {}
        if eval_ds is not None:
            eval_kwargs = dict(
                eval_strategy="steps",
                eval_steps=eval_steps,
                per_device_eval_batch_size=self.train_batch_size,
            )

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            save_strategy="no",
            fp16=False,
            use_cpu=(self.device == "cpu"),
            dataloader_num_workers=0,
            per_device_train_batch_size=self.train_batch_size,
            report_to="none",
            logging_strategy="steps",
            logging_steps=1,
            **eval_kwargs,
        )
```

Return `eval_ds` and `total_steps` alongside the existing tuple. Update the return type and the `_do_train` unpacking accordingly:

```python
        return train_ds, eval_ds, collator, args, labels, total_steps
```

> GLiNER's `Trainer` subclasses HF `Trainer`; passing `eval_dataset` + `eval_strategy="steps"` makes it run the eval loop and log `eval_loss`. If GLiNER's collator needs labels for eval batches (it uses `prepare_labels=True`), the same `collator` is reused — verify a 1-step smoke run logs `eval_loss` (Step 8). If GLiNER rejects `eval_strategy` kwarg name on the installed `transformers` 4.51, use `evaluation_strategy` instead (4.51 accepts `eval_strategy`; keep a fallback try/except in the helper if a smoke run errors).

- [ ] **Step 6: Pass `eval_dataset` into the Trainer + emit eval-step rows**

In `_build_trainer`, accept `eval_ds` and pass it through; extend the `log()` override to tag eval rows. The `_TrackingTrainer` constructor:

```python
        trainer = _TrackingTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            callbacks=[ProgressCallback()],
        )
```

The existing `log()` override already forwards `loss`, `grad_norm`, `learning_rate`, `eval_loss` into a `train_log` event with `step`. That now fires real `eval_loss` on eval steps — no further change needed there. Confirm the method signature change: `_build_trainer(self, model, args, train_ds, eval_ds, collator)`.

- [ ] **Step 7: Emit `total_steps` in `training_start`**

In `_do_train`, update the unpacking and the `training_start` emit:

```python
        train_ds, eval_ds, collator, args, labels, total_steps = self._build_training_components(
            model, cleaned_data, train_data, val_data
        )

        trainer = self._build_trainer(model, args, train_ds, eval_ds, collator)
        ...
        self._emit({
            "type": "training_start",
            "run_id": self.run_id,
            "num_epochs": self.num_epochs,
            "total_steps": total_steps,
        })
```

- [ ] **Step 8: Smoke-test eval streaming**

Run a tiny real training job (≤2 records, 1 epoch) against the dev stack and watch bioner logs for `[ON_LOG FIRED]` lines that include `eval_loss=` and a `train_log` event carrying `eval_loss`. If `eval_loss` never appears, fix the `eval_strategy`/`eval_dataset` wiring before proceeding.

Run (dev): trigger training from the UI or `curl` bioner `/training/start` with a 2-item `training_data` and `val_ratio: 0.5`, then `docker compose logs -f bioner | grep -i eval_loss`.
Expected: at least one `eval_loss` value logged.

- [ ] **Step 9: Lint + commit**

```bash
cd bioner && bash scripts/lint.sh
git add bioner/app/training/gliner_trainer.py bioner/app/tests/test_eval_curve_wiring.py
git commit -m "feat(bioner): eval loop + step-indexed eval-loss streaming + total_steps"
```

---

## Task 9: Backend persists step metrics + total_steps; forwards to UI

The backend `training-events` handler must persist step-indexed train/eval loss (Task 3's `add_step_metric`) and remember `total_steps` for progress. Store `total_steps` on the run (reuse `train_stats` or add a column) and broadcast it.

**Files:**
- Modify: `backend/app/routes/v1/training_events.py` (handle `train_log` with `step`/`eval_loss`; capture `total_steps` from `training_start`)
- Modify: `backend/app/services/training_service.py` (`set_total_steps` helper; persist into `train_stats`)
- Modify: `backend/app/routes/v1/bioner.py` (`GET /bioner/runs/{id}/metrics` returns step + eval_loss)
- Modify: `backend/app/schemas.py` (`TrainingMetricPoint` gains `step`, `eval_loss`)
- Test: `backend/app/tests/test_training_events_step_metrics.py`

**Interfaces:**
- Consumes: `add_step_metric` (Task 3), `train_log`/`training_start` events (Task 8).
- Produces: `GET /bioner/runs/{id}/metrics` → list of `{epoch, loss, step, eval_loss}` ordered by `step` then `epoch`.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/tests/test_training_events_step_metrics.py
def test_train_log_event_persists_step_metric(authed_client, a_pending_run):
    run_id = a_pending_run
    payload = {"type": "train_log", "run_id": run_id, "step": 10,
               "epoch": 1.0, "loss": 0.5}
    r = authed_client.post("/api/v1/bioner/internal/training-events", json=payload)
    assert r.status_code == 200
    metrics = authed_client.get(f"/api/v1/bioner/runs/{run_id}/metrics").json()
    assert any(m["step"] == 10 and m["loss"] == 0.5 for m in metrics)


def test_eval_step_persists_eval_loss(authed_client, a_pending_run):
    run_id = a_pending_run
    payload = {"type": "train_log", "run_id": run_id, "step": 20,
               "epoch": 1.0, "eval_loss": 0.4}
    authed_client.post("/api/v1/bioner/internal/training-events", json=payload)
    metrics = authed_client.get(f"/api/v1/bioner/runs/{run_id}/metrics").json()
    assert any(m["step"] == 20 and m["eval_loss"] == 0.4 for m in metrics)
```

> The internal endpoint may be unauthenticated — if so, post without auth. Mirror existing `training_events` test setup.

- [ ] **Step 2: Run test, verify it fails**

Run: `docker compose exec backend pytest app/tests/test_training_events_step_metrics.py -v`
Expected: FAIL (`train_log` not handled; metrics lack `step`).

- [ ] **Step 3: Handle `train_log` + `training_start` total_steps**

In `training_events.py`, add branches (keep the existing `epoch_update` branch for back-compat with loss-less ticks):

```python
    elif event_type == "training_start":
        total_steps = payload.get("total_steps")
        if total_steps is not None:
            training_service.set_total_steps(db, run_id, int(total_steps))
    elif event_type == "train_log":
        step = payload.get("step")
        if step is not None:
            epoch_raw = payload.get("epoch")
            epoch = int(float(epoch_raw)) if epoch_raw is not None else 0
            training_service.add_step_metric(
                db, run_id, step=int(step), epoch=epoch,
                loss=_safe_float(payload.get("loss")),
                eval_loss=_safe_float(payload.get("eval_loss")),
            )
```

Add `set_total_steps` to `training_service.py`:

```python
def set_total_steps(db: Session, run_id: int, total_steps: int) -> None:
    """Record total optimizer steps on the run (under train_stats) for progress %."""
    run = db.get(TrainingRun, run_id)
    if run is None:
        return
    stats = dict(run.train_stats or {})
    stats["total_steps"] = total_steps
    run.train_stats = stats
    db.add(run)
    db.commit()
```

> SQLModel JSON columns need reassignment (not in-place mutation) to be flagged dirty — `run.train_stats = stats` above does that.

- [ ] **Step 4: Return step + eval_loss from the metrics endpoint**

Update `TrainingMetricPoint` in `schemas.py`:

```python
class TrainingMetricPoint(BaseModel):
    epoch: int
    loss: Optional[float] = None
    step: Optional[int] = None
    eval_loss: Optional[float] = None
```

In `bioner.py`, the `GET /bioner/runs/{id}/metrics` handler must order by `step` (nulls last) then `epoch` and include the new fields. Find the handler (around the runs/metrics route) and ensure it selects all `TrainingMetric` rows for the run and maps them to `TrainingMetricPoint(epoch=..., loss=..., step=..., eval_loss=...)`.

- [ ] **Step 5: Run tests, verify they pass**

Run: `docker compose exec backend pytest app/tests/test_training_events_step_metrics.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/routes/v1/training_events.py backend/app/services/training_service.py backend/app/routes/v1/bioner.py backend/app/schemas.py backend/app/tests/test_training_events_step_metrics.py
git commit -m "feat(backend): persist step-indexed train/eval metrics + total_steps"
```

---

# Phase 4 — Frontend API + types

## Task 10: Frontend API client + types for global model + step metrics

**Files:**
- Modify: `frontend/src/api/monitoring.ts`
- Modify: `frontend/src/types/index.ts`
- Test: `frontend/src/api/__tests__/monitoring.test.ts` (mirror existing api tests if present; otherwise type-level only)

**Interfaces:**
- Produces:
  - `getActiveModel(): Promise<ActiveModelResponse>` → `GET /bioner/active-model`
  - `setActiveModel(modelId: number | null): Promise<ActiveModelResponse>` → `POST /bioner/active-model`
  - `getModelDetail(modelId: number): Promise<ModelDetailResponse>` → `GET /bioner/models/{id}/detail`
  - `TrainingMetric` gains `step?`, `eval_loss?`; `ModelSummary` gains `run_id?`, `is_active`; new `ModelDetailResponse`.
  - **Removed:** `getDatasetActiveModel`, `setDatasetActiveModel`.

- [ ] **Step 1: Update types**

In `frontend/src/types/index.ts`:
- `ModelSummary`: add `run_id?: number | null;` and `is_active?: boolean;`.
- `ActiveModelResponse`: remove `dataset_id`; keep `active_model: ModelSummary | null`.
- `TrainingMetric`: add `step?: number | null;` and `eval_loss?: number | null;`.
- Add:

```ts
export interface ModelDetailResponse {
  model_id: number;
  run_id?: number | null;
  base_model?: string | null;
  train_dataset_ids: number[];
  eval_dataset_ids: number[];
  train_stats?: {
    record_count?: number;
    term_count?: number;
    label_distribution?: Record<string, number>;
    total_steps?: number;
    val_ratio?: number;
  } | null;
  labels: string[];
  per_label_trained: Record<string, Record<string, number>>;
  per_label_baseline: Record<string, Record<string, number>>;
}
```

- [ ] **Step 2: Replace per-dataset model functions with global ones**

In `frontend/src/api/monitoring.ts`, delete `getDatasetActiveModel` and `setDatasetActiveModel`; add:

```ts
/** The GLOBAL active extraction model (null active_model = bioner default). */
export function getActiveModel() {
  return apiRequest<ActiveModelResponse>("/bioner/active-model");
}

/** Set (modelId) or clear (null = default) the GLOBAL active extraction model. */
export function setActiveModel(modelId: number | null) {
  return apiRequest<ActiveModelResponse>("/bioner/active-model", {
    method: "POST",
    body: JSON.stringify({ model_id: modelId }),
  });
}

/** Per-model detail: training datasets, snapshot stats, base-vs-trained eval. */
export function getModelDetail(modelId: number) {
  return apiRequest<ModelDetailResponse>(`/bioner/models/${modelId}/detail`);
}
```

Add `ModelDetailResponse` to the `import type { ... } from "types"` block.

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit`
Expected: errors only at the (not-yet-updated) call sites of the removed functions — those are fixed in Tasks 11–13. Note them; do not fix unrelated files here.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api/monitoring.ts frontend/src/types/index.ts
git commit -m "feat(frontend): global active-model API + model-detail + step metric types"
```

---

# Phase 5 — Frontend Monitor restructure

## Task 11: Rename Comparison→Models, drop page DatasetSelector, build global list + detail

This is the largest frontend task. The existing `ComparisonView.tsx` (~841 lines) is dataset-scoped and full of cross-model widgets. Rather than surgically edit, **create a new `ModelsView.tsx`** and retire `ComparisonView.tsx`. Reuse the existing single-model detail building blocks (loss chart, per-label bars) but drop leaderboard/overlay/heatmap/error-analysis/preferred.

**Files:**
- Create: `frontend/src/pages/Monitor/views/ModelsView.tsx`
- Create: `frontend/src/pages/Monitor/views/ModelsView.module.css`
- Modify: `frontend/src/pages/Monitor/index.tsx` (remove `DatasetSelector`; render `ModelsView`)
- Modify: `frontend/src/pages/Monitor/components/ViewTabs.tsx` (label "Comparison" → "Models")
- Modify: `frontend/src/pages/Monitor/hooks/useMonitor.ts` (rename `MonitorView` `"comparison"` → `"models"`)
- Delete: `frontend/src/pages/Monitor/views/ComparisonView.tsx` (after parity confirmed)
- Modify: `frontend/src/pages/Monitor/components/MonitorProvider.tsx` (provider no longer needs `selectedDatasetId` for the models tab; keep it for Training)
- Test: `frontend/src/pages/Monitor/views/__tests__/ModelsView.test.tsx`

**Interfaces:**
- Consumes: `getModels`, `getModelDetail`, `setActiveModel`, `updateRun`, `deleteRun`, `getRunMetrics`.
- Produces: a global Models tab. Selecting a row loads detail; "Use" sets the global model; Rename/Delete operate per model.

- [ ] **Step 1: Update the view enum + tab label**

In `useMonitor.ts`, change the `MonitorView` type from `"comparison" | "training"` to `"models" | "training"` and update the default `activeView` to `"models"`.

In `ViewTabs.tsx`, change `{ id: "comparison", label: "Comparison" }` to `{ id: "models", label: "Models" }`.

In `Monitor/index.tsx`:
- Remove `import DatasetSelector from "./components/DatasetSelector";` and the `<DatasetSelector />` render.
- Change the conditional to `{activeView === "models" ? <ModelsView /> : <TrainingView />}` and swap the import to `ModelsView`.

> Keep `DatasetSelector.tsx` on disk if `TrainingView` reuses it; otherwise it's now unused — flag for deletion in Step 7.

- [ ] **Step 2: Write the failing test**

```tsx
// frontend/src/pages/Monitor/views/__tests__/ModelsView.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import ModelsView from "../ModelsView";

vi.mock("@api/monitoring", () => ({
  getModels: vi.fn().mockResolvedValue([
    { id: 1, name: "run-1", version: "v", score: 0.8, is_active: true, run_id: 11 },
    { id: 2, name: "run-2", version: "v", score: 0.7, is_active: false, run_id: 12 },
  ]),
  getModelDetail: vi.fn().mockResolvedValue({
    model_id: 1, run_id: 11, train_dataset_ids: [3], eval_dataset_ids: [],
    train_stats: { record_count: 10, term_count: 20, label_distribution: { Drug: 20 } },
    labels: ["Drug"], per_label_trained: { Drug: { exact_f1: 0.8 } },
    per_label_baseline: { Drug: { exact_f1: 0.6 } },
  }),
  setActiveModel: vi.fn(),
  getRunMetrics: vi.fn().mockResolvedValue([]),
  updateRun: vi.fn(),
  deleteRun: vi.fn(),
}));

it("lists models with a Default entry and an active marker", async () => {
  render(<ModelsView />);
  await waitFor(() => expect(screen.getByText("run-1")).toBeInTheDocument());
  expect(screen.getByText(/Default/i)).toBeInTheDocument();   // synthetic default row
  expect(screen.getByText("run-2")).toBeInTheDocument();
});
```

- [ ] **Step 3: Implement `ModelsView.tsx`**

Build a two-pane view: left = global model list (a synthetic "Default (bioner)" row first, then `getModels()` results sorted by `created_at` desc); right = detail for the selected model. Use existing chart wrappers from `frontend/src/components/charts/` (e.g. `LineChart` for the loss curve, `BarChart` for per-label bars — match the props ComparisonView used).

```tsx
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  deleteRun,
  getModelDetail,
  getModels,
  getRunMetrics,
  setActiveModel,
  updateRun,
} from "@api/monitoring";
import type { ModelDetailResponse, ModelSummary, TrainingMetric } from "@/types";
import { useMonitor } from "../hooks/useMonitor";
import styles from "./ModelsView.module.css";

const DEFAULT_ROW: ModelSummary = {
  id: -1,
  name: "Default (bioner)",
  version: "default",
  is_active: false,
};

const ModelsView = () => {
  const { toast } = useMonitor();
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [activeModelId, setActiveModelId] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<ModelDetailResponse | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);

  const reloadModels = useCallback(async () => {
    const list = await getModels();
    setModels(list);
    const active = list.find((m) => m.is_active);
    setActiveModelId(active ? active.id : null); // null => Default is active
  }, []);

  useEffect(() => {
    reloadModels().catch(() => toast.showToast("Failed to load models", "error"));
  }, [reloadModels, toast]);

  // Default row shows active when no trained model is active.
  const rows = useMemo<ModelSummary[]>(
    () => [{ ...DEFAULT_ROW, is_active: activeModelId === null }, ...models],
    [models, activeModelId]
  );

  useEffect(() => {
    if (selectedId == null || selectedId < 0) {
      setDetail(null);
      setMetrics([]);
      return;
    }
    let cancelled = false;
    Promise.all([getModelDetail(selectedId), modelRunMetrics(selectedId)])
      .then(([d, m]) => {
        if (cancelled) return;
        setDetail(d);
        setMetrics(m);
      })
      .catch(() => toast.showToast("Failed to load model detail", "error"));
    return () => {
      cancelled = true;
    };
    // modelRunMetrics resolves the run id from the loaded list
    function modelRunMetrics(id: number): Promise<TrainingMetric[]> {
      const m = models.find((x) => x.id === id);
      return m?.run_id ? getRunMetrics(m.run_id) : Promise.resolve([]);
    }
  }, [selectedId, models, toast]);

  const handleUse = useCallback(
    async (model: ModelSummary) => {
      try {
        const next = model.id < 0 ? null : model.id; // Default => null
        await setActiveModel(next);
        await reloadModels();
        toast.showToast("Active model updated", "success");
      } catch (err) {
        // 409 while extraction is running
        toast.showToast(
          "Cannot change the model while an extraction job is running",
          "error"
        );
      }
    },
    [reloadModels, toast]
  );

  // ... Rename (updateRun by run_id) and Delete (deleteRun by run_id) handlers,
  //     guarded so they are hidden/disabled for the Default row (id < 0).

  return (
    <div className={styles.layout}>
      <ul className={styles.list} role="listbox" aria-label="Trained models">
        {rows.map((m) => (
          <li key={m.id}>
            <button
              type="button"
              className={styles.row}
              aria-selected={selectedId === m.id}
              onClick={() => setSelectedId(m.id)}
            >
              <span className={styles.name}>{m.name}</span>
              {m.score != null && (
                <span className={styles.score}>{(m.score * 100).toFixed(1)}%</span>
              )}
              {m.is_active && <span className={styles.active}>● active</span>}
            </button>
            <button type="button" onClick={() => handleUse(m)}>
              Use
            </button>
            {/* Rename + Delete only when m.id >= 0 */}
          </li>
        ))}
      </ul>

      <section className={styles.detail}>
        {detail ? (
          <ModelDetail detail={detail} metrics={metrics} />
        ) : (
          <p className={styles.placeholder}>
            Select a model to see its training and evaluation detail.
          </p>
        )}
      </section>
    </div>
  );
};

export default ModelsView;
```

`ModelDetail` (same file or a sibling component) renders, for a real model:
- **Training loss curve** from `metrics` (`step` on X, `loss` series; overlay `eval_loss` series where present) using the existing `LineChart` wrapper.
- **Base-vs-trained per-label eval**: a small table/bar from `detail.per_label_baseline` vs `detail.per_label_trained` (label, base exact_f1, trained exact_f1, Δ).
- **Trained on**: `detail.train_dataset_ids` (+ eval ids if any).
- **Training-time stats**: `detail.train_stats.record_count`, `term_count`.
- **Label coverage**: `detail.labels` with `detail.train_stats.label_distribution` counts.

For the Default row (`id < 0`), `ModelDetail` shows a short "Untrained default model — no training history." message (no detail fetch happens since `selectedId < 0` short-circuits).

- [ ] **Step 4: Run the view test, verify it passes**

Run: `cd frontend && npm run test -- src/pages/Monitor/views/__tests__/ModelsView.test.tsx`
Expected: PASS.

- [ ] **Step 5: Delete `ComparisonView.tsx` + dead imports**

Once `ModelsView` covers the per-model detail, delete `views/ComparisonView.tsx`. Grep for remaining imports: `cd frontend && grep -rn "ComparisonView\|getAllRunEvaluations\|getAllEvaluations\|getRunErrorAnalysis\|ModelComparisonHeatmap" src/` — remove now-unused imports/exports. The heatmap component (`charts/ModelComparisonHeatmap.tsx`), `getAllEvaluations`/`getAllRunEvaluations`, and `getRunErrorAnalysis` are no longer used by the UI; delete the dead frontend chart component and leave the now-unused API helpers only if other code references them (grep first).

- [ ] **Step 6: Typecheck + lint**

Run: `cd frontend && npx tsc -b --noEmit && npm run lint`
Expected: no errors. Fix any fallout in `MonitorProvider`/`useMonitor` from the enum rename.

- [ ] **Step 7: Commit**

```bash
git add -A frontend/src/pages/Monitor
git commit -m "feat(frontend): global Models tab (list + per-model detail); remove Comparison + cross-model widgets"
```

---

# Phase 6 — Frontend Training tab

## Task 12: Curated GLiNER baselines + step/% progress + train+eval loss plot

**Files:**
- Modify: `frontend/src/pages/Monitor/views/TrainingView.tsx`
- Modify: `frontend/src/pages/Monitor/charts/TrainingLossChart.tsx` (overlay eval-loss series)
- Modify: `frontend/src/pages/Monitor/components/MonitorProvider.tsx` (progress from `step/total_steps`; capture `eval_loss`)
- Test: `frontend/src/pages/Monitor/views/__tests__/TrainingView.test.tsx`

**Interfaces:**
- Consumes: `training_start` (now with `total_steps`) and `train_log` (with `step`, `loss`, `eval_loss`) WS messages.
- Produces: baseline `<select>` with curated options + custom path; progress text `step / total (pct%)`; loss chart with train + eval series.

- [ ] **Step 1: Replace the base-model selector with a curated dropdown**

In `TrainingView.tsx`, replace the default/custom radio (`DEFAULT_MODEL = "urchade/gliner_small-v2.1"`) with:

```tsx
const GLINER_BASELINES = [
  { value: "urchade/gliner_multi-v2.1", label: "Multilingual (default)" },
  { value: "urchade/gliner_large-v2.1", label: "Large (best, slower)" },
  {
    value: "E3-JSI/gliner-multi-med-ner-synthetic-v1",
    label: "Biomedical / clinical (multilingual)",
  },
] as const;
const DEFAULT_MODEL = GLINER_BASELINES[0].value;
```

Render a `<select>` over `GLINER_BASELINES` plus a final `"custom"` option. When `"custom"` is chosen, reveal a text input for the HF id / local path and show a warning:

```tsx
{useCustomModel && (
  <p className={styles.warning} role="alert">
    ⚠ Advanced: custom base models must be GLiNER-compatible. An incompatible
    model will fail to train.
  </p>
)}
```

Keep the existing `customModel`/`useCustomModel` provider state; `resolvedModel` stays `useCustomModel ? customModel : selectedBaseline`.

- [ ] **Step 2: Step-based progress in the provider**

In `MonitorProvider.tsx`, the WS handler currently derives `progress` from epochs. Update:
- On `training_start`: store `totalSteps` from `msg.total_steps`.
- On `train_log`: set `currentStep = msg.step`; `progress = totalSteps ? Math.min(100, Math.round((currentStep / totalSteps) * 100)) : progress`. Append `{ step, loss, eval_loss }` to `trainingMetrics` (keep existing append shape; add `eval_loss`).

Expose `currentStep` and `totalSteps` from the provider for the progress label.

- [ ] **Step 3: Progress label + eval overlay**

In `TrainingView.tsx` progress section, render `Step {currentStep} / {totalSteps} ({progress}%)` alongside the existing progress bar.

In `TrainingLossChart.tsx`, accept the metrics that may include `eval_loss` and render a second series (eval) over the same X (`step`), styled distinctly (e.g. `CHART.relaxedF1` color), so divergence between train and eval loss is visible. Train series uses `loss`; eval series filters to points where `eval_loss != null`.

- [ ] **Step 4: Write/extend the test**

```tsx
// frontend/src/pages/Monitor/views/__tests__/TrainingView.test.tsx
it("offers the curated GLiNER baselines with multilingual default", () => {
  render(/* TrainingView within a mocked MonitorProvider */);
  const select = screen.getByLabelText(/base model/i);
  expect(select).toHaveValue("urchade/gliner_multi-v2.1");
  expect(screen.getByText(/Biomedical \/ clinical/i)).toBeInTheDocument();
});

it("shows step/total progress while training", () => {
  // provider mocked with isTraining=true, currentStep=40, totalSteps=200, progress=20
  render(/* ... */);
  expect(screen.getByText(/40 \/ 200 \(20%\)/)).toBeInTheDocument();
});
```

- [ ] **Step 5: Run tests, typecheck, lint**

Run: `cd frontend && npm run test -- src/pages/Monitor/views/__tests__/TrainingView.test.tsx && npx tsc -b --noEmit && npm run lint`
Expected: PASS / no errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/Monitor
git commit -m "feat(frontend): curated GLiNER baselines + step/% progress + train/eval loss plot"
```

---

# Phase 7 — Frontend Term Extraction

## Task 13: Read-only active-model chip + Monitor link

**Files:**
- Modify: `frontend/src/pages/DatasetTermExtraction/index.tsx`
- Modify: `frontend/src/pages/DatasetTermExtraction/styles.module.css` (chip styles)
- Test: `frontend/src/pages/DatasetTermExtraction/__tests__/activeModelChip.test.tsx`

**Interfaces:**
- Consumes: `getActiveModel` (Task 10).
- Produces: a read-only tag/chip in the stats row showing the global active model name + a link to `/monitor`.

- [ ] **Step 1: Swap the model state + fetch**

In `index.tsx`:
- Change imports: remove `getDatasetActiveModel`, `getModels`, `setDatasetActiveModel`; add `getActiveModel`.
- Replace the `models`/`activeModelId` state with a single `activeModelName` state:

```tsx
  // Read-only display of the GLOBAL active extraction model (selected in Monitor).
  const [activeModelName, setActiveModelName] = useState<string>("Default model");
```

- Replace the fetch effect:

```tsx
  useEffect(() => {
    let cancelled = false;
    getActiveModel()
      .then((res) => {
        if (cancelled) return;
        setActiveModelName(res.active_model?.name ?? "Default model");
      })
      .catch(() => {
        /* optional; ignore */
      });
    return () => {
      cancelled = true;
    };
  }, []);
```

- Delete `modelOptions`, `hasTrainedModels`, and `handleSelectModel`.

- [ ] **Step 2: Replace the `<Select>` with a chip + link**

Where the model `<Select>` rendered in the stats row, render:

```tsx
  <span className={styles.modelChip} title="Extraction model (set in Monitor)">
    Model: {activeModelName}
    <Link to="/monitor" className={styles.modelChipLink}>
      Monitor
    </Link>
  </span>
```

Ensure `Link` is imported from `react-router-dom` (used elsewhere in the app). Add `.modelChip` / `.modelChipLink` styles to `styles.module.css` as a pill/tag (rounded background, small font, inline-flex) consistent with existing chip styling in the codebase (grep `border-radius` chips for reference).

- [ ] **Step 3: Write the test**

```tsx
// frontend/src/pages/DatasetTermExtraction/__tests__/activeModelChip.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

vi.mock("@api/monitoring", () => ({
  getActiveModel: vi.fn().mockResolvedValue({ active_model: { id: 1, name: "run-7" } }),
}));

it("renders the active model name as a chip linking to Monitor", async () => {
  // render the stats-row fragment (or the page with required providers/mocks)
  await waitFor(() => expect(screen.getByText(/run-7/)).toBeInTheDocument());
  expect(screen.getByRole("link", { name: /Monitor/i })).toHaveAttribute("href", "/monitor");
});
```

> If rendering the whole page is heavy, extract the chip into a tiny `ActiveModelChip` component and test that in isolation.

- [ ] **Step 4: Run test, typecheck, lint**

Run: `cd frontend && npm run test -- src/pages/DatasetTermExtraction && npx tsc -b --noEmit && npm run lint`
Expected: PASS / no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/DatasetTermExtraction
git commit -m "feat(frontend): read-only active-model chip in Term Extraction, linking to Monitor"
```

---

# Phase 8 — End-to-end verification

## Task 14: Full-stack smoke test

**Files:** none (verification only).

- [ ] **Step 1: Rebuild + migrate**

```bash
docker-compose up -d --build
docker compose exec backend alembic upgrade head
docker compose exec backend alembic current   # expect 009 (head)
```

- [ ] **Step 2: Backend test suite**

Run: `docker compose exec backend pytest -q`
Expected: green (fix any residual references to the removed per-dataset active-model endpoints / `Dataset.active_model_id`). Grep backend for stragglers: `grep -rn "active_model_id\|active-model\|resolve_active_model" backend/app` — every hit should match the new global design.

- [ ] **Step 3: Frontend build + tests**

Run: `cd frontend && npm run build && npm run test`
Expected: build succeeds; tests pass.

- [ ] **Step 4: Manual click-through (against Docker frontend)**

Verify each acceptance criterion:
1. Term Extraction shows **Model: <name>** as a chip with a Monitor link; **no dropdown**.
2. Monitor has **Models** + **Training** tabs and **no page-level dataset selector**.
3. Models tab lists trained runs + a **Default (bioner)** row; **no** leaderboard/overlay/heatmap; selecting a model shows loss curve, base-vs-trained eval, datasets, training-time stats, label coverage; **Use** sets the global model (reflected back in Term Extraction); Rename/Delete work; **no preferred-star, no error-analysis panel**.
4. Start an extraction job, then try to change the model in Monitor → **409 / blocked** with a clear message.
5. Training tab: baseline dropdown defaults to **Multilingual**, lists Large + Biomedical + Custom (custom shows the ⚠ warning); starting a run shows **step / total (pct%)** and a loss plot with **both train and eval** curves; the eval curve lets you see divergence.

- [ ] **Step 5: Lint sweep + final commit**

```bash
ruff check backend bioner
cd frontend && npm run lint
git add -A && git commit -m "chore: lint + e2e verification for global-model/monitor restructure"
```

---

## Self-Review notes (carried into execution)

- **Spec coverage:** model-selection-in-Monitor (Tasks 5,11), state-current-model-in-extraction (Task 13), global model (Tasks 1,4,5,6), training not locked to dataset (already multi-dataset; preserved), model list + per-model eval/datasets/stats/coverage (Task 7,11), no cross-model comparison (Task 11), training tab dataset+hyperparams+baseline (Task 12), progress iterations+%+loss+eval plot (Tasks 8,9,12). ✓
- **Open implementation confirmations (do during execution, not blocking):**
  1. `ExtractionJob` active-status enum values (Task 4 Step 1).
  2. `training_service.get_dataset_ids` existence/signature (Task 7 Step 4).
  3. The multi-dataset stats service function name for the `train_stats` snapshot (Task 6 Step 5).
  4. GLiNER `Trainer` accepts `eval_strategy="steps"` on the installed transformers 4.51 (Task 8 Step 5/8 smoke test); fall back to `evaluation_strategy` if needed.
  5. Frontend chart wrapper props for `LineChart`/`BarChart` (match how the old `ComparisonView` passed series).
- **Tenancy note:** the global model list is intentionally instance-wide (any user may select any trained model). Per-owner guards, if ever wanted, belong on delete/rename, not the list.
