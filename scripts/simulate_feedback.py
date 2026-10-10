"""Simulate user feedback and reviewer decisions for testing the retraining loop.

Usage:
    python scripts/simulate_feedback.py --n 12
"""

import argparse
import json
import urllib.request

import pandas as pd

API_URL = "http://localhost:8000"
CONTROL_URL = "http://localhost:8001"
ADMIN_TOKEN = "dev-admin-token"


def http_post(url: str, data: dict, headers: dict | None = None) -> dict:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate feedback items and reviews")
    parser.add_argument("--n", type=int, default=12, help="Number of feedback items to create")
    parser.add_argument("--auto-approve", action="store_true", default=True,
                        help="Automatically approve simulated feedback items")
    args = parser.parse_args()

    test_df = pd.read_parquet("data/processed/test.parquet")
    samples = test_df.sample(args.n, random_state=42)

    created = 0
    for _, row in samples.iterrows():
        text = str(row["text"])
        true_label = "positive" if int(row["label"]) == 1 else "negative"

        # 1. Get prediction
        pred = http_post(f"{API_URL}/predict", {"text": text})
        pred_id = pred.get("id")
        if not pred_id:
            continue

        # 2. Submit user feedback with the true ground truth label
        fb = http_post(f"{API_URL}/feedback", {"prediction_id": pred_id, "label": true_label})
        fb_id = fb["feedback_id"]

        # 3. Approve in control center if auto-approve
        if args.auto_approve:
            http_post(
                f"{CONTROL_URL}/control/feedback/{fb_id}/review",
                {"decision": "approve", "label": true_label},
                headers={"X-Admin-Token": ADMIN_TOKEN},
            )
        created += 1

    print(f"Created and approved {created} feedback items successfully.")


if __name__ == "__main__":
    main()
