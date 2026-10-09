from sentiment.gates import evaluate_gates
from sentiment.prepare import clean

GATES = {"min_accuracy": 0.88, "min_macro_f1": 0.88, "min_class_recall": 0.85,
         "max_regression": 0.01}
GOOD = {"accuracy": 0.90, "macro_f1": 0.90, "min_class_recall": 0.89}


def test_good_model_passes():
    assert evaluate_gates(GOOD, GATES).passed


def test_low_accuracy_fails():
    r = evaluate_gates({**GOOD, "accuracy": 0.80}, GATES)
    assert not r.passed
    assert [c["name"] for c in r.checks if not c["passed"]] == ["accuracy"]


def test_imbalanced_recall_fails():
    assert not evaluate_gates({**GOOD, "min_class_recall": 0.5}, GATES).passed


def test_regression_vs_champion_blocks():
    assert not evaluate_gates(GOOD, GATES, champion_f1=0.95).passed


def test_small_regression_allowed():
    assert evaluate_gates(GOOD, GATES, champion_f1=0.905).passed


def test_clean_strips_html_and_whitespace():
    assert clean("Great<br /><br />movie   !") == "Great movie !"
