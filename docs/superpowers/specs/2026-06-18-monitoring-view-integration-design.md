# Monitoring-view Integration — Design

**Date:** 2026-06-18
**Branch:** `feat/monitoring-integration` (based on `ninakokalj/main`)
**Status:** Design — awaiting review

## Goal

Bring the GLiNER **training pipeline + monitoring dashboard** from the
`Monitoring-view` branch into `main`, reconciled onto the database schema
introduced by `ninakokalj/main`. The backend must follow the style already
present in the codebase. Frontend UI is updated in a later pass; this work keeps
API response shapes stable so the UI change is minimal.

## Scope

**In scope (this work):**
- Monitoring dashboard backend (counts, runs, evaluations, live training feed).
- GLiNER training pipeline (start/stop, per-epoch metrics, final evaluation,
  trained-model artifact), persisting into ninakokalj's schema.
- `ner_metrics.py` (exact + relaxed F1) brought into `backend/` and `bioner/`.

**Deferred (future, not this work):**
- Bucket C — model-management / model-settings UI, `UserModelPreference`.
- Bucket D — label-linking (`SourceTermLink` + `LinkArrowOverlay`), storybook
  scaffolding, bioner Docker/requirements churn, DatasetOverview/Upload reworks.
- The frontend UI polish pass (separate task). Response shapes are kept stable
  here so that pass is small.

## Context: why reconciliation is needed

`ninakokalj/main` and `Monitoring-view` independently model the same domain
(models / training / evaluation) with **incompatible** tables.

- ninakokalj built the **extraction side**: `Model`, `SourceTermEx`,
  `ExtractionJob.model_id` / `currently_used` are fully implemented in
  `bioner.py`. `Evaluation` and `ModelTrainRecordLink` are **defined but unused**
  — scaffolding for a training/eval flow ninakokalj never built.
- Monitoring-view built the **training side** with its own parallel tables:
  `TrainingRun`, `TrainingMetric`, `TrainingEvaluation`, `ModelArtifact`,
  `UserModelPreference`, `SourceTermLink`.

They are complementary halves. The integration makes `Model` the shared anchor,
routes training outputs into ninakokalj's `Model` / `Evaluation` /
`ModelTrainRecordLink`, and adds only the genuinely-missing pieces (run lifecycle
+ per-epoch metric).

Field-level findings that drive the design:
- The Monitor UI only charts `epoch` + `loss` per epoch (per-epoch
  precision/recall/f1 are written but never read).
- The UI reads evaluation as `per_label: {label: {…metrics…}}` only.
- `ModelArtifact`'s columns are written but never read.
- The UI actually calls `/bioner/*` endpoints; the branch's separate
  `monitoring.py` router is dead code.

## Decisions (locked with user)

1. **Scope:** monitoring dashboard + training pipeline, reconciled to ninakokalj.
   C/D deferred.
2. **Base branch:** `ninakokalj/main` merges into `main` first; this branch is
   developed on top of `ninakokalj/main`.
3. **Schema strategy (Approach 2):** dedicated `training_run` owns lifecycle and
   produces a `Model` on success; reuse ninakokalj's `Evaluation` and
   `ModelTrainRecordLink`; drop `TrainingEvaluation` and `ModelArtifact`.
4. **`Evaluation.score`** is a flexible JSON object holding multiple metric
   types — at minimum **exact F1** and **relaxed F1**, plus precision/recall and
   any others. Computed via `ner_metrics.py`.
5. **Per-epoch table** keeps only `epoch` + `loss`.
6. **Routing:** training/monitoring endpoints stay under `/bioner` (matching the
   current frontend). The branch's duplicate `monitoring.py` router is removed;
   logic lives in services, `bioner.py` is a thin route layer.

## Schema

### Reused from ninakokalj (no change)

```python
class Evaluation(SQLModel, table=True):       # final eval, ONE ROW PER LABEL
    __tablename__ = "evaluation"
    id: Optional[int] = Field(default=None, primary_key=True)
    label: str                                # e.g. "Diagnosis", or "micro avg"
    score: Dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
                                              # {exact_f1, relaxed_f1, precision, recall, ...}
    dataset_id: int = Field(foreign_key="dataset.id", ondelete="CASCADE", nullable=False)
    model_id: int   = Field(foreign_key="model.id",   ondelete="CASCADE", nullable=False)
    # relationships unchanged

class ModelTrainRecordLink(SQLModel, table=True):   # records used to train a model
    model_id:  Optional[int] = Field(default=None, foreign_key="model.id",  primary_key=True)
    record_id: Optional[int] = Field(default=None, foreign_key="record.id", primary_key=True)
```

### Extended: `Model` becomes the trained artifact

`Model` currently is `(id, name, version)` and is used by the extraction side
(`SourceTermEx.model_id`, `ExtractionJob.model_id`, `currently_used`). Extend it
so a trained model produced by the pipeline is a `Model` row — keeping a single
clean artifact list.

```python
class Model(SQLModel, table=True):
    __tablename__ = "model"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    version: str
    base_model: Optional[str] = Field(default=None)          # NEW: e.g. urchade/gliner_small-v2.1
    path: Optional[str] = Field(default=None)                # NEW: saved artifact location
    dataset_id: Optional[int] = Field(                       # NEW: training dataset
        default=None, foreign_key="dataset.id", ondelete="SET NULL")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # NEW
    # existing relationships: source_terms, train_records, evaluations
    training_run: Optional["TrainingRun"] = Relationship(back_populates="model")  # NEW
```

### New: `training_run` (lifecycle — ninakokalj has no equivalent)

```python
class TrainingRun(SQLModel, table=True):
    __tablename__ = "training_run"
    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id", ondelete="CASCADE",
                            nullable=False, index=True)
    base_model: str
    labels: List[str] = Field(sa_column=Column(JSON))
    val_ratio: float = Field(default=0.0)                    # train/eval split
    status: str = Field(default="pending", index=True)       # pending|running|completed|failed|stopped
    error_message: Optional[str] = Field(default=None)       # mirrors Dataset.error_message
    model_id: Optional[int] = Field(default=None, foreign_key="model.id",
                                    ondelete="SET NULL")      # set on success
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    metrics: list["TrainingMetric"] = Relationship(
        back_populates="run",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"})
    model: Optional["Model"] = Relationship(back_populates="training_run")
```

### New: `training_metric` (live curve — only epoch + loss)

```python
class TrainingMetric(SQLModel, table=True):
    __tablename__ = "training_metric"
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="training_run.id", ondelete="CASCADE",
                        nullable=False, index=True)
    epoch: int
    loss: Optional[float] = Field(default=None)
    run: Optional["TrainingRun"] = Relationship(back_populates="metrics")
```

### Dropped / Deferred

- **Dropped:** `TrainingEvaluation` (→ `Evaluation`), `ModelArtifact`
  (path → `Model.path`; metrics → `Evaluation`).
- **Deferred:** `UserModelPreference`, `SourceTermLink`.

### Column-placement rationale

- Artifact path on `Model.path` — the artifact owns its location.
- Final metrics in `Evaluation` keyed by `model_id`; a run reaches them via
  `run.model_id`.
- `val_ratio`, `error_message` on `training_run` (config / failure state).

## Training lifecycle (reconciled)

1. **Start** — `POST /bioner/training/start` `{dataset_id, labels, base_model,
   val_ratio}` → create `TrainingRun(status="pending")`, call bioner trainer,
   return `{run_id}`.
2. **Running** — trainer event `training_info` → `status="running"`.
3. **Per-epoch** — trainer event `epoch_update` `{epoch, loss}` → insert
   `TrainingMetric(run_id, epoch, loss)`; broadcast over websocket.
4. **Evaluation** — trainer event `evaluation_completed` `{per_label, …}` → on
   success a `Model` row is created (if not already), then **one `Evaluation`
   row per label** with `score = {exact_f1, relaxed_f1, precision, recall, …}`,
   `model_id`, `dataset_id`. Recompute via `ner_metrics.py` if raw spans are
   provided.
5. **Model saved** — trainer event `completed` `{output_path}` → ensure `Model`
   row (`name`, `version`, `base_model`, `path=output_path`, `dataset_id`,
   `created_at`), set `run.model_id`, populate `ModelTrainRecordLink` for the
   training records, set `status="completed"`.
6. **Stop / error** — `stopped` → `status="stopped"`; `error` → `status="failed"`,
   `error_message` set.

## Backend components

- **`backend/app/models_db.py`** — schema changes above.
- **Alembic migration** — additive: extend `model`, create `training_run`,
  `training_metric`. (`evaluation`, `model_train_record_link` already created by
  ninakokalj's migrations.) Manual autogenerate + review per repo convention.
- **`backend/app/routes/v1/bioner.py`** — thin routes:
  `POST /training/start`, `POST /training/stop/{run_id}`,
  `GET /datasets/{id}/full-stats`, `GET /datasets/{id}/runs`,
  `GET /datasets/{id}/runs/evaluations`, `GET /runs/{id}/evaluation`,
  `GET /evaluations`, `WS /ws/training`. Remove duplicate `monitoring.py` router.
- **Services (logic):**
  - `services/training_service.py` — run lifecycle state machine, event handling.
  - `services/evaluation_service.py` — assemble/store `Evaluation` rows, compute
    exact/relaxed F1 via `ner_metrics.py`, build `per_label` response.
  - `services/websocket_manager.py` + `services/bioner_client.py` — live feed +
    trainer client (ported from Monitoring-view).
- **`backend/app/library/ner_metrics.py`** + **`bioner/app/library/ner_metrics.py`**
  — exact/relaxed F1 (ported from Monitoring-view).
- **`bioner/app/training/*`** — GLiNER trainer, job manager, callbacks (ported;
  event payloads aligned to the lifecycle above).
- **`backend/app/schemas.py`** — `GLiNERTrainingRequest` and monitoring response
  schemas; response shapes kept stable (`per_label: {label: {metrics}}`,
  run/full-stats shapes).

## API response shapes (kept stable for the UI)

- `GET /bioner/datasets/{id}/full-stats` → `{totalRecords, totalTerms,
  labelDistribution: {label: count}}`.
- `GET /bioner/datasets/{id}/runs` → `[{run_id, status}]`.
- `GET /bioner/runs/{id}/evaluation` → `{run_id, per_label: {label: {exact_f1,
  relaxed_f1, precision, recall}}}`. (UI gains exact/relaxed in its later pass;
  existing keys remain present.)
- `POST /bioner/training/start` → `{run_id}`. `POST /bioner/training/stop/{id}`
  → `{message}`.
- `WS /bioner/ws/training` — event types: `training_info`, `epoch_update`,
  `train_log`, `completed`, `stopped`, `error`.

## Frontend (this work = minimal)

No UI redesign here. Only what's required for the backend to be exercised:
- `frontend/src/api/monitoring.ts` paths confirmed against `/bioner/*`.
- Existing Monitor page kept functional against stable shapes.
Full UI polish + exact/relaxed F1 rendering is the deferred follow-up task.

## Testing

- **Unit:** `ner_metrics` exact vs relaxed F1 on known spans;
  `evaluation_service` per_label assembly from `Evaluation` rows;
  `training_service` status transitions (pending→running→completed / failed /
  stopped).
- **Integration (pytest + FastAPI TestClient):** start → epoch events → eval
  event → completed; assert `TrainingRun`, `TrainingMetric`, `Model`,
  `Evaluation`, `ModelTrainRecordLink` rows; assert `GET` endpoints return the
  documented shapes.
- **Migration:** `alembic upgrade head` then `downgrade -1` round-trips on a
  scratch DB.

## Risks / open points

- ninakokalj/main is not yet in `main`; if it is later squash-merged, this
  branch may need a rebase before its own PR.
- `Model` now serves both extraction and training; ensure trainer-created rows
  don't disrupt extraction's `currently_used` logic (a trained `Model` only
  becomes "currently used" through the existing extraction flow, never
  automatically on training completion).
- bioner trainer event payloads must be aligned to the reconciled lifecycle;
  any field the trainer emits but we no longer store is dropped intentionally.
