"""Evaluate the trained model on the hold-out test set and apply quality gates."""

import json

import joblib
import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, recall_score

from sentiment.config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR, load_params, setup_mlflow
from sentiment.gates import evaluate_gates


def champion_f1(client: MlflowClient, model_name: str) -> float | None:
    """Macro-F1 of the model currently aliased `champion`, if any."""
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
    except Exception:
        return None
    return client.get_run(mv.run_id).data.metrics.get("test_macro_f1")


def main() -> None:
    params = load_params()
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    model = joblib.load(MODEL_DIR / "model.joblib")
    run_id = json.loads((MODEL_DIR / "run_info.json").read_text())["run_id"]

    preds = model.predict(test["text"])
    metrics = {
        "accuracy": float(accuracy_score(test["label"], preds)),
        "macro_f1": float(f1_score(test["label"], preds, average="macro")),
        "min_class_recall": float(recall_score(test["label"], preds, average=None).min()),
        "n_test": len(test),
    }

    setup_mlflow()
    client = MlflowClient()
    gate = evaluate_gates(metrics, params["gates"],
                          champion_f1(client, params["registry"]["model_name"]))

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORT_DIR / "gates.json").write_text(json.dumps(gate.to_dict(), indent=2))

    with mlflow.start_run(run_id=run_id):
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("gates_passed", float(gate.passed))
        mlflow.log_artifact(str(REPORT_DIR / "gates.json"), artifact_path="evaluation")

    print(json.dumps(metrics, indent=2))
    for c in gate.checks:
        print(("PASS" if c["passed"] else "FAIL"), c["name"], c["value"], c["op"], c["threshold"])


if __name__ == "__main__":
    main()
