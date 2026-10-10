"""Train the TF-IDF + Logistic Regression baseline and track it in MLflow."""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from sentiment.config import FEEDBACK_DIR, MODEL_DIR, PROCESSED_DIR, load_params, setup_mlflow


def build_pipeline(p: dict) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, p["ngram_max"]), min_df=p["min_df"],
                                      max_features=p["max_features"], sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=p["C"])),
        ]
    )


def train_model(params: dict | None = None, feedback_path: Path | None = None) -> str:
    """Train pipeline on base dataset + reviewed feedback, log to MLflow, save model."""
    if params is None:
        params = load_params()["train"]
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    n_base = len(train)
    n_feedback = 0

    fb_file = feedback_path or (FEEDBACK_DIR / "reviewed.parquet")
    if fb_file.exists():
        try:
            fb_df = pd.read_parquet(fb_file)
            if not fb_df.empty and {"text", "label"} <= set(fb_df.columns):
                fb_clean = fb_df[["text", "label"]].dropna()
                n_feedback = len(fb_clean)
                if n_feedback > 0:
                    train = pd.concat([train, fb_clean], ignore_index=True).drop_duplicates(
                        subset=["text"], keep="last"
                    )
        except Exception as e:
            print(f"Warning: could not load feedback from {fb_file}: {e}")

    setup_mlflow()
    with mlflow.start_run(run_name="baseline-tfidf-logreg") as run:
        mlflow.log_params(params)
        mlflow.log_param("n_base_train", n_base)
        mlflow.log_param("n_feedback", n_feedback)
        mlflow.log_param("n_train_total", len(train))
        mlflow.log_artifact(str(PROCESSED_DIR / "stats.json"), artifact_path="data")
        if fb_file.exists() and n_feedback > 0:
            mlflow.log_artifact(str(fb_file), artifact_path="data")

        pipe = build_pipeline(params)
        pipe.fit(train["text"], train["label"])
        preds = pipe.predict(train["text"])
        mlflow.log_metric("train_accuracy", float(accuracy_score(train["label"], preds)))
        mlflow.log_metric("train_macro_f1",
                          float(f1_score(train["label"], preds, average="macro")))
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(pipe, MODEL_DIR / "model.joblib")
        (MODEL_DIR / "run_info.json").write_text(json.dumps({"run_id": run.info.run_id}))
        print(f"Trained (base={n_base}, feedback={n_feedback}, total={len(train)}). "
              f"MLflow run: {run.info.run_id}")
        return run.info.run_id


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
