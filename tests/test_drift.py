import pandas as pd

from monitoring.drift import parse_columns, summarize
from monitoring.features import FEATURE_COLUMNS, text_features


def test_text_features_basic():
    f = text_features(pd.Series(["Great movie!!", "ok", None]))
    assert list(f.columns) == FEATURE_COLUMNS
    assert f.loc[0, "exclamation_count"] == 2
    assert f.loc[0, "word_count"] == 2
    assert f.loc[1, "text_length"] == 2
    assert f.loc[2, "text_length"] == 0  # missing text must not crash


def _metric(col, method, value, thr=0.05):
    return {"metric_name": f"ValueDrift(column={col},method={method},threshold={thr})",
            "value": value}


def test_parse_columns_p_value_vs_distance():
    report = {"metrics": [
        {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 1}},
        _metric("a", "K-S p_value", 0.001),            # small p -> drifted
        _metric("b", "K-S p_value", 0.5),              # large p -> fine
        _metric("c", "Wasserstein distance (normed)", 0.4, 0.1),   # large distance -> drifted
        _metric("d", "Wasserstein distance (normed)", 0.01, 0.1),
    ]}
    cols = {c["column"]: c["drifted"] for c in parse_columns(report)}
    assert cols == {"a": True, "b": False, "c": True, "d": False}


def test_summarize_threshold():
    cols = [{"column": c, "drifted": d} for c, d in
            [("a", True), ("b", True), ("c", False), ("d", False), ("e", False)]]
    assert summarize(cols, 100, 50, 0.4)["status"] == "drift"      # 2/5 = 0.4 >= 0.4
    assert summarize(cols, 100, 50, 0.5)["status"] == "ok"
    s = summarize(cols, 100, 50, 0.4)
    assert s["drifted_columns"] == ["a", "b"] and s["drift_share"] == 0.4
