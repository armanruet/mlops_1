"""Train the TF-IDF + Logistic Regression baseline and track it in MLflow."""

import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from sentiment.config import MODEL_DIR, PROCESSED_DIR, load_params, setup_mlflow


def build_pipeline(p: dict) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, p["ngram_max"]), min_df=p["min_df"],
                                      max_features=p["max_features"], sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=p["C"])),
        ]
    )


def main() -> None:
    params = load_params()["train"]
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")

    setup_mlflow()
    with mlflow.start_run(run_name="baseline-tfidf-logreg") as run:
        mlflow.log_params(params)
        mlflow.log_param("n_train", len(train))
        mlflow.log_artifact(str(PROCESSED_DIR / "stats.json"), artifact_path="data")

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
        print(f"Trained. MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
