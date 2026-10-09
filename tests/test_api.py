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
