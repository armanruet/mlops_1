"""Shared config: params and MLflow setup."""

from pathlib import Path

import mlflow
import yaml

MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT = "imdb-sentiment"
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


def load_params(path: str = "params.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
