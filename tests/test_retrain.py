import pytest
from fastapi.testclient import TestClient

import control.main as control_main
from control.main import app


@pytest.fixture
def control_client():
    with TestClient(app) as c:
        yield c


def test_control_feedback_list(control_client, monkeypatch):
    sample = [
        {
            "id": 1,
            "prediction_id": 10,
            "text": "Great movie",
            "predicted_label": "positive",
            "confidence": 0.9,
            "model_version": "v1",
            "user_label": "positive",
            "reviewed_label": None,
            "review_status": "pending",
            "created_at": "2026-10-10T00:00:00Z",
            "reviewed_at": None,
        }
    ]
    monkeypatch.setattr(control_main, "list_feedback", lambda status, limit: sample)
    r = control_client.get("/control/feedback?status=pending")
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert r.json()[0]["id"] == 1


def test_control_review_requires_token(control_client, monkeypatch):
    monkeypatch.setattr(control_main, "ADMIN_TOKEN", "secret")
    r = control_client.post("/control/feedback/1/review", json={"decision": "approve"})
    assert r.status_code == 403


def test_control_review_success(control_client, monkeypatch):
    monkeypatch.setattr(control_main, "ADMIN_TOKEN", "secret")
    monkeypatch.setattr(
        control_main,
        "review_feedback",
        lambda fid, decision, label, reviewer: {
            "id": fid,
            "review_status": "approved",
            "reviewed_label": label or "positive",
        },
    )
    r = control_client.post(
        "/control/feedback/1/review",
        json={"decision": "approve", "label": "positive"},
        headers={"X-Admin-Token": "secret"},
    )
    assert r.status_code == 200
    assert r.json()["review_status"] == "approved"


def test_control_retrain_requires_token(control_client, monkeypatch):
    monkeypatch.setattr(control_main, "ADMIN_TOKEN", "secret")
    r = control_client.post("/control/retrain", json={"force": True})
    assert r.status_code == 403


def test_control_retrain_blocked_under_threshold(control_client, monkeypatch):
    monkeypatch.setattr(control_main, "ADMIN_TOKEN", "secret")
    monkeypatch.setattr(
        control_main,
        "run_retraining",
        lambda min_reviewed=None, force=False: {
            "ok": False,
            "reason": "Insufficient approved feedback: 2 items, minimum 10 required",
        },
    )
    r = control_client.post(
        "/control/retrain", json={"force": False}, headers={"X-Admin-Token": "secret"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "Insufficient" in body["reason"]


def test_control_retrain_success(control_client, monkeypatch):
    monkeypatch.setattr(control_main, "ADMIN_TOKEN", "secret")
    monkeypatch.setattr(
        control_main,
        "run_retraining",
        lambda min_reviewed=None, force=False: {
            "ok": True,
            "run_id": "fake_run_123",
            "candidate_registered": True,
        },
    )
    r = control_client.post(
        "/control/retrain", json={"force": True}, headers={"X-Admin-Token": "secret"}
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["run_id"] == "fake_run_123"
