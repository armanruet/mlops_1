"""Load and validate the IMDB dataset."""

from pathlib import Path

import pandas as pd

LABELS = {0: "negative", 1: "positive"}
RAW_DIR = Path("data/raw")


def download(raw_dir: Path = RAW_DIR) -> None:
    """Download IMDB from Hugging Face and store train/test as parquet."""
    from datasets import load_dataset

    raw_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("stanfordnlp/imdb")
    for split in ("train", "test"):
        ds[split].to_pandas().to_parquet(raw_dir / f"{split}.parquet", index=False)


def load_split(split: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_parquet(raw_dir / f"{split}.parquet")


def validate(df: pd.DataFrame) -> None:
    """Basic data quality checks; raises ValueError on failure."""
    if not {"text", "label"} <= set(df.columns):
        raise ValueError("Expected columns: text, label")
    if df["text"].isna().any() or (df["text"].str.strip() == "").any():
        raise ValueError("Empty or null review text found")
    if not set(df["label"].unique()) <= set(LABELS):
        raise ValueError("Unexpected label values")


if __name__ == "__main__":
    download()
    for s in ("train", "test"):
        validate(load_split(s))
    print("Downloaded and validated IMDB.")
