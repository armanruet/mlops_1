"""Train the TF-IDF + Logistic Regression baseline."""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from sentiment.data import load_split, validate

MODEL_DIR = Path("models")


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                                      sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=4.0)),
        ]
    )


def main() -> None:
    train, test = load_split("train"), load_split("test")
    validate(train)
    validate(test)

    pipe = build_pipeline()
    pipe.fit(train["text"], train["label"])
    preds = pipe.predict(test["text"])

    metrics = {
        "accuracy": float(accuracy_score(test["label"], preds)),
        "macro_f1": float(f1_score(test["label"], preds, average="macro")),
        "n_train": len(train),
        "n_test": len(test),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()
