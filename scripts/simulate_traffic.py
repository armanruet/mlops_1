"""Send simulated traffic to the inference API.

    python scripts/simulate_traffic.py --n 200            # normal IMDB-like reviews
    python scripts/simulate_traffic.py --n 200 --shift    # drifted: short, shouting, "!!!"
"""

import argparse
import json
import urllib.request

import pandas as pd


def shift(text: str) -> str:
    """Make a review look unlike the training data: first sentence only, shouted."""
    first = text.split(".")[0][:120]
    return first.upper() + "!!!"


def post(url: str, text: str) -> dict:
    req = urllib.request.Request(
        f"{url}/predict", data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--shift", action="store_true")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    test = pd.read_parquet("data/processed/test.parquet")
    texts = test.sample(a.n, random_state=a.seed)["text"].tolist()
    counts: dict[str, int] = {}
    for t in texts:
        r = post(a.url, shift(t) if a.shift else t)
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"Sent {a.n} {'shifted' if a.shift else 'normal'} requests: {counts}")


if __name__ == "__main__":
    main()
