import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import api.main as api_main
from api.main import app, state


@pytest.fixture
def client():
    texts = ["great movie loved it", "wonderful acting", "terrible waste of time", "awful boring"]
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipe.fit(texts, [1, 1, 0, 0])
    with TestClient(app) as c:
        state["model"] = pipe
        state["info"] = {"version": "7"}
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    assert r.json()["model_version"] == "v7"


def test_predict(client):
    r = client.post("/predict", json={"text": "great wonderful movie"})
    body = r.json()
    assert r.status_code == 200
    assert body["label"] == "positive"
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["model_version"] == "v7"


def test_predict_rejects_empty(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_reload_requires_token(client, monkeypatch):
    monkeypatch.setattr(api_main, "ADMIN_TOKEN", "secret")
    assert client.post("/admin/reload").status_code == 403
    assert client.post("/admin/reload", headers={"X-Admin-Token": "bad"}).status_code == 403


def test_reload_disabled_without_configured_token(client, monkeypatch):
    monkeypatch.setattr(api_main, "ADMIN_TOKEN", "")
    assert client.post("/admin/reload", headers={"X-Admin-Token": ""}).status_code == 403


def test_reload_picks_up_new_export(client, monkeypatch, tmp_path):
    import json

    import joblib

    joblib.dump(state["model"], tmp_path / "model.joblib")
    (tmp_path / "serving.json").write_text(json.dumps({"version": "9"}))
    monkeypatch.setattr(api_main, "ADMIN_TOKEN", "secret")
    monkeypatch.setattr(api_main, "SERVING_DIR", tmp_path)
    r = client.post("/admin/reload", headers={"X-Admin-Token": "secret"})
    assert r.status_code == 200 and r.json()["model_version"] == "v9"
    assert client.get("/model-info").json()["version"] == "9"
    assert 'model_info{version="v9"} 1.0' in client.get("/metrics").text


def test_metrics_exposed_after_predict(client):
    client.post("/predict", json={"text": "great wonderful movie"})
    body = client.get("/metrics").text
    assert 'predictions_total{label="positive",model_version="v7"}' in body
    assert 'http_requests_total{method="POST",path="/predict",status="200"}' in body
    assert "prediction_confidence_bucket" in body


def test_predict_logs_prediction(client, monkeypatch):
    logged = []
    monkeypatch.setattr(api_main, "log_prediction", lambda *a: logged.append(a))
    client.post("/predict", json={"text": "great wonderful movie"})
    assert len(logged) == 1
    text, label, confidence, version = logged[0]
    assert (text, label, version) == ("great wonderful movie", "positive", "v7")
    assert 0.5 <= confidence <= 1.0


def test_predict_survives_logging_failure(client, monkeypatch):
    from api import db

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://x:y@localhost:1/none")
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "create_engine", lambda *a, **k: boom())
    # log_prediction swallows the failure, so the user still gets a prediction
    assert client.post("/predict", json={"text": "great wonderful movie"}).status_code == 200
