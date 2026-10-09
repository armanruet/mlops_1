"""Quality gates: pure functions deciding whether a model may be approved."""

from dataclasses import dataclass, field


@dataclass
class GateResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks}


def _check(name: str, value: float, threshold: float, op: str) -> dict:
    ok = value >= threshold if op == ">=" else value <= threshold
    return {"name": name, "value": round(value, 5), "threshold": threshold, "op": op, "passed": ok}


def evaluate_gates(metrics: dict, gates: dict, champion_f1: float | None = None) -> GateResult:
    """Apply absolute thresholds and, if a champion exists, a no-regression check."""
    checks = [
        _check("accuracy", metrics["accuracy"], gates["min_accuracy"], ">="),
        _check("macro_f1", metrics["macro_f1"], gates["min_macro_f1"], ">="),
        _check("min_class_recall", metrics["min_class_recall"], gates["min_class_recall"], ">="),
    ]
    if champion_f1 is not None:
        checks.append(
            _check("f1_vs_champion", metrics["macro_f1"], champion_f1 - gates["max_regression"],
                   ">=")
        )
    return GateResult(passed=all(c["passed"] for c in checks), checks=checks)
