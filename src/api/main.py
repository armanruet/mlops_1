"""FastAPI inference service. Serves the exported `champion` model."""

import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from api.db import log_prediction, submit_feedback
from sentiment.data import LABELS

SERVING_DIR = Path(os.getenv("SERVING_DIR", "models/serving"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # empty disables admin endpoints
MAX_CHARS = 10_000

state: dict = {}

# --- Prometheus metrics -------------------------------------------------------
HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
PREDICTIONS = Counter("predictions_total", "Predictions served", ["label", "model_version"])
CONFIDENCE = Histogram(
    "prediction_confidence", "Confidence of the predicted class",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
INPUT_CHARS = Histogram(
    "input_length_chars", "Length of classified text",
    buckets=(50, 100, 250, 500, 1000, 2000, 4000, 10000),
)
MODEL_INFO = Gauge("model_info", "Currently served model version", ["version"])


def load_serving_model() -> bool:
    """(Re)load model + metadata from SERVING_DIR. Keeps the old model on failure."""
    model_file = SERVING_DIR / "model.joblib"
    if not model_file.exists():
        return False
    model = joblib.load(model_file)
    info_file = SERVING_DIR / "serving.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    state["model"], state["info"] = model, info
    MODEL_INFO.clear()
    MODEL_INFO.labels(version=model_version()).set(1)
    return True


def model_version() -> str:
    version = state.get("info", {}).get("version")
    return f"v{version}" if version else "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_serving_model()
    yield
    state.clear()


app = FastAPI(title="Sentiment API", version="0.3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"],
                   allow_headers=["*"])


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        # Use the route template (not the raw URL) to keep label cardinality bounded.
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
        HTTP_LATENCY.labels(path).observe(time.perf_counter() - start)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


class PredictResponse(BaseModel):
    id: int | None = None
    label: str
    confidence: float
    model_version: str


class FeedbackRequest(BaseModel):
    prediction_id: int
    label: str = Field(pattern="^(positive|negative)$")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "model_version": model_version()}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info")
def model_info():
    return {"loaded": "model" in state, **state.get("info", {})}


@app.post("/admin/reload")
def reload_model(x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not load_serving_model():
        raise HTTPException(status_code=404, detail="No exported model found")
    return {"reloaded": True, "model_version": model_version()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, background: BackgroundTasks):
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be blank")
    probs = model.predict_proba([req.text])[0]
    idx = int(probs.argmax())
    label = LABELS[int(model.classes_[idx])]
    confidence, version = float(probs[idx]), model_version()

    PREDICTIONS.labels(label, version).inc()
    CONFIDENCE.observe(confidence)
    INPUT_CHARS.observe(len(req.text))
    pred_id = log_prediction(req.text, label, confidence, version)
    return PredictResponse(id=pred_id, label=label, confidence=confidence, model_version=version)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    try:
        fb_id = submit_feedback(req.prediction_id, req.label)
        return {"ok": True, "feedback_id": fb_id}
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

