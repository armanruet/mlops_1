"""FastAPI inference service."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentiment.data import LABELS

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "baseline-v1")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
MAX_CHARS = 10_000

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Sentiment API", version="0.1.0", lifespan=lifespan)
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
    return {"status": "ok", "model_loaded": "model" in state, "model_version": MODEL_VERSION}


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
                           model_version=MODEL_VERSION)
