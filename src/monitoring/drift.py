"""Evidently data/prediction drift: recent production traffic vs a held-out reference sample.

Usage: python -m monitoring.drift
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from api.db import recent_predictions
from monitoring.features import FEATURE_COLUMNS, text_features
from sentiment.config import PROCESSED_DIR, REPORT_DIR, SERVING_DIR, load_params

DRIFT_DIR = REPORT_DIR / "drift"
NUMERIC = [*FEATURE_COLUMNS, "confidence"]
CATEGORICAL = ["predicted_label"]
_METRIC = re.compile(r"ValueDrift\(column=(?P<col>[^,]+),method=(?P<method>[^,]+),"
                     r"threshold=(?P<thr>[\d.eE+-]+)\)")


def featurize(texts: pd.Series, labels: pd.Series, confidences: pd.Series) -> pd.DataFrame:
    df = text_features(texts)
    df["confidence"] = confidences.reset_index(drop=True).astype(float)
    df["predicted_label"] = labels.reset_index(drop=True).astype(str)
    return df


def build_reference(model, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Held-out test sample scored by the serving model (what 'normal' looks like)."""
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    sample = test.sample(min(n, len(test)), random_state=seed)
    probs = model.predict_proba(sample["text"])
    labels = pd.Series(probs.argmax(axis=1)).map({0: "negative", 1: "positive"})
    return featurize(sample["text"], labels, pd.Series(probs.max(axis=1)))


def parse_columns(report_dict: dict) -> list[dict]:
    """Extract per-column drift verdicts from an Evidently report dict."""
    cols = []
    for m in report_dict["metrics"]:
        match = _METRIC.match(m["metric_name"])
        if not match:
            continue
        value, thr = float(m["value"]), float(match["thr"])
        # p-value tests drift when small; distance metrics drift when large.
        drifted = value < thr if "p_value" in match["method"] else value > thr
        cols.append({"column": match["col"], "method": match["method"], "value": value,
                     "threshold": thr, "drifted": bool(drifted)})
    return cols


def summarize(
    columns: list[dict], n_reference: int, n_current: int, share_threshold: float
) -> dict:
    drifted = [c["column"] for c in columns if c["drifted"]]
    share = len(drifted) / len(columns) if columns else 0.0
    detected = share >= share_threshold
    return {
        "status": "drift" if detected else "ok",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_reference": n_reference,
        "n_current": n_current,
        "drift_share": round(share, 3),
        "drift_share_threshold": share_threshold,
        "drifted_columns": drifted,
        "columns": columns,
        "recommended_action": (
            "Review recent traffic and consider retraining" if detected else "No action needed"
        ),
    }


def latest_summary() -> dict | None:
    path = DRIFT_DIR / "latest.json"
    return json.loads(path.read_text()) if path.exists() else None


def run_drift(serving_dir: Path = SERVING_DIR) -> dict:
    cfg = load_params()["monitoring"]
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    current_raw = recent_predictions(limit=cfg["window"])
    if len(current_raw) < cfg["min_samples"]:
        summary = {"status": "insufficient_data", "n_current": len(current_raw),
                   "min_samples": cfg["min_samples"],
                   "generated_at": datetime.now(UTC).isoformat(timespec="seconds")}
    else:
        model = joblib.load(serving_dir / "model.joblib")
        reference = build_reference(model, n=cfg["reference_size"])
        current = featurize(current_raw["text"], current_raw["predicted_label"],
                            current_raw["confidence"])
        definition = DataDefinition(numerical_columns=NUMERIC, categorical_columns=CATEGORICAL)
        report = Report([DataDriftPreset(drift_share=cfg["drift_share_threshold"])])
        snapshot = report.run(Dataset.from_pandas(current, data_definition=definition),
                              Dataset.from_pandas(reference, data_definition=definition))
        snapshot.save_html(str(DRIFT_DIR / "latest.html"))
        summary = summarize(parse_columns(snapshot.dict()), len(reference), len(current),
                            cfg["drift_share_threshold"])

    (DRIFT_DIR / "latest.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run_drift(), indent=2))
