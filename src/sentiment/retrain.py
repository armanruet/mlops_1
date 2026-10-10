"""Retraining pipeline: export approved feedback, train candidate, evaluate gates, register.

Usage:
    python -m sentiment.retrain export
    python -m sentiment.retrain run [--force] [--min-reviewed N]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from api.db import approved_feedback
from sentiment.config import FEEDBACK_DIR, load_params
from sentiment.evaluate import evaluate_model
from sentiment.registry import register
from sentiment.train import train_model

logger = logging.getLogger("sentiment.retrain")


def export_feedback(output_file: Path = FEEDBACK_DIR / "reviewed.parquet") -> pd.DataFrame:
    """Export all human-approved feedback from the database to a versioned parquet dataset."""
    try:
        df = approved_feedback()
    except Exception as e:
        logger.warning("Database not accessible (%s); creating empty feedback dataset", e)
        df = pd.DataFrame(
            columns=[
                "feedback_id",
                "text",
                "label",
                "predicted_label",
                "model_version",
                "reviewed_at",
            ]
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"Exported {len(df)} approved feedback items to {output_file}")
    return df


def run_retraining(min_reviewed: int | None = None, force: bool = False) -> dict:
    """Execute the full retraining lifecycle.

    1. Checks reviewed feedback count against the configured quality threshold.
    2. Exports approved feedback to data/feedback/reviewed.parquet.
    3. Retrains TF-IDF + LogisticRegression baseline with combined data.
    4. Evaluates against the hold-out test set and current champion.
    5. Registers a new candidate in the MLflow Model Registry if gates pass.
    """
    params = load_params()
    retrain_cfg = params.get("retraining", {})
    threshold = (
        min_reviewed if min_reviewed is not None else retrain_cfg.get("min_reviewed", 10)
    )

    try:
        df = export_feedback()
    except Exception as e:
        logger.exception("Failed to export feedback from database")
        return {"ok": False, "reason": f"Database export error: {e}"}

    if len(df) < threshold and not force:
        return {
            "ok": False,
            "reason": (
                f"Insufficient approved feedback: {len(df)} items available, "
                f"minimum {threshold} required (use force=True to override)."
            ),
            "n_feedback": len(df),
            "threshold": threshold,
        }

    run_id = train_model()
    metrics, gates = evaluate_model()
    exit_code = register()

    return {
        "ok": True,
        "run_id": run_id,
        "n_feedback": len(df),
        "threshold": threshold,
        "metrics": metrics,
        "gates": gates,
        "candidate_registered": exit_code == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Model retraining workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("export", help="Export approved feedback to parquet")

    run_parser = subparsers.add_parser("run", help="Run full retraining pipeline")
    run_parser.add_argument("--force", action="store_true", help="Retrain regardless of threshold")
    run_parser.add_argument("--min-reviewed", type=int, default=None,
                            help="Override min reviewed threshold")

    args = parser.parse_args()

    if args.command == "export":
        export_feedback()
        return 0

    if args.command == "run":
        result = run_retraining(min_reviewed=args.min_reviewed, force=args.force)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
