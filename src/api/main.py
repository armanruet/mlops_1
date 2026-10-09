"""FastAPI inference service. Serves the exported `champion` model."""

import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentiment.data import LABELS

SERVING_DIR = Path(os.getenv("SERVING_DIR", "models/serving"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # empty disables admin endpoints
MAX_CHARS = 10_000

state: dict = {}


def load_serving_model() -> bool:
    """(Re)load model + metadata from SERVING_DIR. Keeps the old model on failure."""
    model_file = SERVING_DIR / "model.joblib"
    if not model_file.exists():
        return False
    model = joblib.load(model_file)
    info_file = SERVING_DIR / "serving.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    state["model"], state["info"] = model, info
    return True


def model_version() -> str:
    version = state.get("info", {}).get("version")
    return f"v{version}" if version else "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_serving_model()
    yield
    state.clear()


app = FastAPI(title="Sentiment API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"],
                   allow_headers=["*"])


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "model_version": model_version()}


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
def predict(req: PredictRequest):
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be blank")
    probs = model.predict_proba([req.text])[0]
    idx = int(probs.argmax())
    return PredictResponse(label=LABELS[int(model.classes_[idx])], confidence=float(probs[idx]),
                           model_version=model_version())
