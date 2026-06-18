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


def test_run_evaluation_shape(client):
    from app.services import training_service

    db = client.session
    run = training_service.create_run(
        db,
        dataset_id=client.dataset_id,
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
