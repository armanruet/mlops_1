import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from api.main import app, state


@pytest.fixture
def client():
    texts = ["great movie loved it", "wonderful acting", "terrible waste of time", "awful boring"]
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipe.fit(texts, [1, 1, 0, 0])
    with TestClient(app) as c:
        state["model"] = pipe
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict(client):
    r = client.post("/predict", json={"text": "great wonderful movie"})
    body = r.json()
    assert r.status_code == 200
    assert body["label"] == "positive"
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_rejects_empty(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "   "}).status_code == 422
