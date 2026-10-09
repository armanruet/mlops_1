"""Download stage + prepare stage (clean, validate, profile)."""

import json
import re
import sys

from sentiment.config import PROCESSED_DIR
from sentiment.data import RAW_DIR, download, load_split, validate

HTML_BR = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", HTML_BR.sub(" ", text)).strip()


def run_download() -> None:
    download(RAW_DIR)


def run_prepare() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ("train", "test"):
        df = load_split(split, RAW_DIR)
        validate(df)
        before = len(df)
        df = df.assign(text=df["text"].map(clean)).drop_duplicates(subset="text")
        validate(df)
        df.to_parquet(PROCESSED_DIR / f"{split}.parquet", index=False)
        stats[split] = {
            "rows": len(df),
            "duplicates_removed": before - len(df),
            "class_balance": {str(k): int(v) for k, v in df["label"].value_counts().items()},
            "mean_chars": round(float(df["text"].str.len().mean()), 1),
        }
    (PROCESSED_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    {"download": run_download, "prepare": run_prepare}[sys.argv[1]]()
