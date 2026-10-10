"""Shared config: params and MLflow setup."""

import os
from pathlib import Path

import mlflow
import yaml

# Tracking/registry lives on the MLflow server started by docker compose.
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT = "imdb-sentiment"
PROCESSED_DIR = Path("data/processed")
FEEDBACK_DIR = Path("data/feedback")
MODEL_DIR = Path("models")
SERVING_DIR = MODEL_DIR / "serving"
REPORT_DIR = Path("reports")


def load_params(path: str = "params.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
