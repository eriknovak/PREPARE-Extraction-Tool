"""Route tests for the training/monitoring endpoints under /bioner.

These run in Docker (the full app imports heavy ML deps). They use the FastAPI
TestClient against the real app with dependency overrides for the DB session
(in-memory SQLite) and the current user (a fake user).
"""

import os

os.environ.setdefault("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
os.environ.setdefault("ELASTICSEARCH_URL", "http://localhost:9200")
os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("JWT_SECRET_KEY", "test")

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.core.database import get_session
from app.main import app
from app.models_db import Dataset, Record, User
from app.routes.v1.auth import get_current_user


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        user = User(username="t", hashed_password="h")
        session.add(user)
        session.commit()
        session.refresh(user)

        ds = Dataset(name="ds", labels=["Drug"], user_id=user.id)
        session.add(ds)
        session.commit()
        session.refresh(ds)

        rec = Record(
            patient_id="p",
            visit_date=datetime.now(timezone.utc),
            text="aspirin",
            dataset_id=ds.id,
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)

        def _get_session_override():
            yield session

        app.dependency_overrides[get_session] = _get_session_override
        app.dependency_overrides[get_current_user] = lambda: user
        c = TestClient(app)
        c.dataset_id = ds.id
        c.run_record_id = rec.id
        c.session = session
        yield c
        app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _no_trainer(monkeypatch):
    monkeypatch.setattr(
        "app.services.bioner_client.start_training", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "app.services.bioner_client.stop_training", lambda *a, **k: None
    )


def test_start_creates_run(client):
    resp = client.post(
        "/api/v1/bioner/training/start",
        json={
            "dataset_id": client.dataset_id,
            "labels": ["Drug"],
            "base_model": "urchade/gliner_small-v2.1",
            "val_ratio": 0.1,
        },
    )
    assert resp.status_code == 200
    assert "run_id" in resp.json()


def test_full_stats_shape(client):
    resp = client.get(f"/api/v1/bioner/datasets/{client.dataset_id}/full-stats")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"totalRecords", "totalTerms", "labelDistribution"}
    assert body["totalRecords"] >= 1


def test_full_stats_multi_shape(client):
    resp = client.post(
        "/api/v1/bioner/datasets/full-stats",
        json={"dataset_ids": [client.dataset_id]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"totalRecords", "totalTerms", "labelDistribution"}
    assert body["totalRecords"] >= 1


def test_start_multi_dataset_records_links(client):
    from app.models_db import Dataset
    from app.services import training_service

    db = client.session
    primary = db.get(Dataset, client.dataset_id)
    ds2 = Dataset(name="ds2", labels=["Drug"], user_id=primary.user_id)
    db.add(ds2)
    db.commit()
    db.refresh(ds2)

    resp = client.post(
        "/api/v1/bioner/training/start",
        json={
            "dataset_ids": [client.dataset_id, ds2.id],
            "labels": ["Drug"],
            "base_model": "urchade/gliner_small-v2.1",
            "val_ratio": 0.1,
        },
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    assert set(training_service.get_dataset_ids(db, run_id, role="train")) == {
        client.dataset_id,
        ds2.id,
    }


def test_start_rejects_unowned_dataset(client):
    resp = client.post(
        "/api/v1/bioner/training/start",
        json={
            "dataset_ids": [999999],
            "labels": ["Drug"],
            "base_model": "urchade/gliner_small-v2.1",
            "val_ratio": 0.1,
        },
    )
    assert resp.status_code == 404


def test_run_evaluation_shape(client):
    from app.services import training_service

    db = client.session
    run = training_service.create_run(
        db,
        dataset_ids=[client.dataset_id],
        base_model="b",
        labels=["Drug"],
        val_ratio=0.0,
    )
    training_service.record_evaluation(
        db, run.id, {"Drug": {"exact_f1": 0.8, "relaxed_f1": 0.9}}
    )
    resp = client.get(f"/api/v1/bioner/runs/{run.id}/evaluation")
    assert resp.status_code == 200
    assert resp.json()["per_label"]["Drug"]["exact_f1"] == 0.8


def test_start_rejects_empty_labels(client):
    resp = client.post(
        "/api/v1/bioner/training/start",
        json={
            "dataset_id": client.dataset_id,
            "labels": [],
            "base_model": "urchade/gliner_small-v2.1",
            "val_ratio": 0.1,
        },
    )
    assert resp.status_code == 422


def test_start_rejects_bad_val_ratio(client):
    resp = client.post(
        "/api/v1/bioner/training/start",
        json={
            "dataset_id": client.dataset_id,
            "labels": ["Drug"],
            "base_model": "urchade/gliner_small-v2.1",
            "val_ratio": 1.0,
        },
    )
    assert resp.status_code == 422


def test_start_rejects_concurrent_active_run(client):
    payload = {
        "dataset_id": client.dataset_id,
        "labels": ["Drug"],
        "base_model": "urchade/gliner_small-v2.1",
        "val_ratio": 0.1,
    }
    first = client.post("/api/v1/bioner/training/start", json=payload)
    assert first.status_code == 200
    second = client.post("/api/v1/bioner/training/start", json=payload)
    assert second.status_code == 409


def test_list_runs_paginated_and_enriched(client):
    from app.services import training_service

    db = client.session
    run = training_service.create_run(
        db,
        dataset_ids=[client.dataset_id],
        base_model="b",
        labels=["Drug"],
        val_ratio=0.2,
    )
    training_service.record_evaluation(
        db, run.id, {"Drug": {"exact_f1": 0.8}, "Dose": {"exact_f1": 0.6}}
    )
    resp = client.get(f"/api/v1/bioner/datasets/{client.dataset_id}/runs?page=1&limit=20")
    assert resp.status_code == 200
    body = resp.json()
    assert "runs" in body and "pagination" in body
    assert body["pagination"]["total"] >= 1
    summary = next(r for r in body["runs"] if r["run_id"] == run.id)
    assert summary["base_model"] == "b"
    assert summary["labels"] == ["Drug"]
    assert summary["val_ratio"] == 0.2
    # macro-F1 = mean(0.8, 0.6)
    assert abs(summary["score"] - 0.7) < 1e-6


def test_rename_and_prefer_run(client):
    from app.services import training_service

    db = client.session
    run = training_service.create_run(
        db,
        dataset_ids=[client.dataset_id],
        base_model="b",
        labels=["Drug"],
        val_ratio=0.0,
    )
    resp = client.patch(
        f"/api/v1/bioner/runs/{run.id}",
        json={"name": "My best run", "preferred": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "My best run"
    assert body["preferred"] is True


def test_delete_run(client):
    from app.services import training_service

    db = client.session
    run = training_service.create_run(
        db,
        dataset_ids=[client.dataset_id],
        base_model="b",
        labels=["Drug"],
        val_ratio=0.0,
    )
    resp = client.delete(f"/api/v1/bioner/runs/{run.id}")
    assert resp.status_code == 200
    follow = client.get(f"/api/v1/bioner/runs/{run.id}/evaluation")
    # evaluation endpoint still 200s but the run is gone; list no longer contains it
    listing = client.get(f"/api/v1/bioner/datasets/{client.dataset_id}/runs")
    assert all(r["run_id"] != run.id for r in listing.json()["runs"])
    assert follow.status_code == 200
