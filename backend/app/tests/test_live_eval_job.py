"""Tests for the live-eval background job.

The worker runs a trained model over a dataset's held-out reviewed records,
writes the model's predictions to ``SourceTermEx`` (never ``SourceTerm``), and
stores precision/recall/F1 metrics on the ``LiveEvalJob`` row. The NER HTTP call
and the bioner model hot-swap are mocked; runs against in-memory SQLite.
"""

import os

os.environ.setdefault("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
os.environ.setdefault("ELASTICSEARCH_URL", "http://localhost:9200")
os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("JWT_SECRET_KEY", "test")

from datetime import datetime, timezone

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import app.routes.v1.live_eval as live_eval
from app.models_db import (
    AppSettings,
    Dataset,
    LiveEvalJob,
    Model,
    ModelTrainRecordLink,
    Record,
    SourceTerm,
    SourceTermEx,
    User,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture
def engine(monkeypatch):
    """Fresh in-memory DB wired into the live_eval module's module-level engine."""
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(live_eval, "engine", eng)
    yield eng
    SQLModel.metadata.drop_all(eng)


def _mk_record(s, ds_id, text, reviewed=True):
    r = Record(
        patient_id="p",
        visit_date=datetime.now(timezone.utc),
        text=text,
        dataset_id=ds_id,
        reviewed=reviewed,
    )
    s.add(r)
    s.commit()
    s.refresh(r)
    return r


def _gold(s, record_id, value, label, start, end):
    s.add(
        SourceTerm(
            record_id=record_id,
            value=value,
            label=label,
            start_position=start,
            end_position=end,
            automatically_extracted=False,
        )
    )
    s.commit()


def _seed(engine):
    """Dataset with a held-out reviewed record (gold), a trained record, an
    unreviewed record, and a reviewed record without gold offsets. Returns
    ids for the eval model, a previously-active model, and the held-out record.
    """
    with Session(engine) as s:
        user = User(username="t", hashed_password="h")
        s.add(user)
        s.commit()
        s.refresh(user)

        ds = Dataset(name="ds", labels=["Drug"], user_id=user.id)
        eval_model = Model(name="eval", version="1", path="/models/eval")
        prev_model = Model(name="prev", version="1", path="/models/prev")
        s.add(ds)
        s.add(eval_model)
        s.add(prev_model)
        s.commit()
        s.refresh(ds)
        s.refresh(eval_model)
        s.refresh(prev_model)

        # The globally-active model (must be restored after eval).
        s.add(AppSettings(id=1, active_model_id=prev_model.id))
        s.commit()

        # Held-out reviewed record with two gold Drug terms.
        heldout = _mk_record(s, ds.id, "aspirin and ibuprofen", reviewed=True)
        _gold(s, heldout.id, "aspirin", "Drug", 0, 7)
        _gold(s, heldout.id, "ibuprofen", "Drug", 12, 21)

        # Trained record (in the model's train_records) — must be excluded.
        trained = _mk_record(s, ds.id, "paracetamol", reviewed=True)
        _gold(s, trained.id, "paracetamol", "Drug", 0, 11)
        s.add(ModelTrainRecordLink(model_id=eval_model.id, record_id=trained.id))
        s.commit()

        # Unreviewed record — excluded.
        unreviewed = _mk_record(s, ds.id, "morphine", reviewed=False)
        _gold(s, unreviewed.id, "morphine", "Drug", 0, 8)

        # Reviewed but no gold offsets — excluded.
        no_gold = _mk_record(s, ds.id, "codeine", reviewed=True)

        return {
            "dataset_id": ds.id,
            "eval_model_id": eval_model.id,
            "prev_model_id": prev_model.id,
            "prev_path": prev_model.path,
            "heldout_id": heldout.id,
            "trained_id": trained.id,
            "unreviewed_id": unreviewed.id,
            "no_gold_id": no_gold.id,
        }


def _make_job(engine, dataset_id, model_id):
    with Session(engine) as s:
        job = LiveEvalJob(
            dataset_id=dataset_id,
            model_id=model_id,
            total=0,
            completed=0,
            status="pending",
            currently_used=True,
        )
        s.add(job)
        s.commit()
        s.refresh(job)
        return job.id


def test_heldout_excludes_trained_unreviewed_and_ungolded(engine):
    ids = _seed(engine)
    with Session(engine) as s:
        heldout = live_eval._heldout_record_ids(s, ids["eval_model_id"], ids["dataset_id"])
    assert heldout == [ids["heldout_id"]]


def test_worker_writes_only_source_term_ex_and_computes_metrics(engine, monkeypatch):
    ids = _seed(engine)

    # NER predicts one correct term (aspirin) and one wrong term (tylenol),
    # missing ibuprofen: tp=1, fp=1, fn=1 -> precision=recall=f1=0.5.
    def fake_post(url, json=None, timeout=None):
        return _FakeResponse(
            [
                {"text": "aspirin", "label": "Drug", "start": 0, "end": 7, "score": 0.9},
                {"text": "tylenol", "label": "Drug", "start": 30, "end": 37, "score": 0.8},
            ]
        )

    activate_calls = []
    monkeypatch.setattr(live_eval.requests, "post", fake_post)
    monkeypatch.setattr(live_eval, "_activate_on_bioner", lambda path: activate_calls.append(path))

    job_id = _make_job(engine, ids["dataset_id"], ids["eval_model_id"])
    live_eval.run_live_eval_job(
        job_id=job_id,
        model_id=ids["eval_model_id"],
        dataset_id=ids["dataset_id"],
        labels=["Drug"],
    )

    with Session(engine) as s:
        job = s.get(LiveEvalJob, job_id)
        assert job.status == "completed"
        assert job.total == 1 and job.completed == 1

        # Metrics: per-label exact/relaxed/overlap + macro aggregate.
        m = job.metrics
        assert m["heldout_count"] == 1
        assert m["labels"] == ["Drug"]
        exact = m["per_label"]["Drug"]["exact"]
        assert exact["precision"] == pytest.approx(0.5)
        assert exact["recall"] == pytest.approx(0.5)
        assert exact["f1"] == pytest.approx(0.5)
        assert m["aggregate"]["exact"]["f1"] == pytest.approx(0.5)
        assert m["gold_entity_count"] == 2
        assert m["pred_entity_count"] == 2

        # Predictions written to SourceTermEx with the eval model_id, only for
        # the held-out record.
        ex = s.exec(select(SourceTermEx)).all()
        assert {e.value for e in ex} == {"aspirin", "tylenol"}
        assert all(e.model_id == ids["eval_model_id"] for e in ex)
        assert all(e.record_id == ids["heldout_id"] for e in ex)

        # SourceTerm (gold) is never written by live eval: still exactly the 4
        # gold terms seeded (2 held-out + trained + unreviewed).
        gold = s.exec(select(SourceTerm)).all()
        assert len(gold) == 4

    # bioner hot-swapped to the eval model path, then restored to the previously
    # active model path.
    assert activate_calls == ["/models/eval", ids["prev_path"]]


def test_worker_restores_active_model_on_ner_failure(engine, monkeypatch):
    ids = _seed(engine)

    class _Boom(live_eval.requests.RequestException):
        pass

    def fake_post(url, json=None, timeout=None):
        raise _Boom("bioner down")

    activate_calls = []
    monkeypatch.setattr(live_eval.requests, "post", fake_post)
    monkeypatch.setattr(live_eval, "_activate_on_bioner", lambda path: activate_calls.append(path))

    job_id = _make_job(engine, ids["dataset_id"], ids["eval_model_id"])
    live_eval.run_live_eval_job(
        job_id=job_id,
        model_id=ids["eval_model_id"],
        dataset_id=ids["dataset_id"],
        labels=["Drug"],
    )

    with Session(engine) as s:
        job = s.get(LiveEvalJob, job_id)
        assert job.status == "failed"
        assert job.error_message

    # Even on failure, the previously-active model is restored in the finally block.
    assert activate_calls == ["/models/eval", ids["prev_path"]]
