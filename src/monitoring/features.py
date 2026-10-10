"""Input features used for drift monitoring (cheap, model-agnostic text statistics)."""

import pandas as pd

FEATURE_COLUMNS = [
    "text_length",
    "word_count",
    "avg_word_length",
    "exclamation_count",
    "uppercase_ratio",
]


def text_features(texts: pd.Series) -> pd.DataFrame:
    s = texts.fillna("").astype(str).reset_index(drop=True)
    length = s.str.len()
    words = s.str.split().str.len()
    return pd.DataFrame(
        {
            "text_length": length,
            "word_count": words,
            "avg_word_length": length / words.clip(lower=1),
            "exclamation_count": s.str.count("!"),
            "uppercase_ratio": s.str.count(r"[A-Z]") / length.clip(lower=1),
        }
    )
