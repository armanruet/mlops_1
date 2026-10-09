"""MLflow Model Registry: register candidates, promote to champion, export for serving.

Aliases: `candidate` = passed gates, awaiting approval. `champion` = serving model.
Usage:
    python -m sentiment.registry register
    python -m sentiment.registry promote [--version N]
    python -m sentiment.registry export      # write champion to models/serving/
    python -m sentiment.registry status
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sentiment.config import MODEL_DIR, REPORT_DIR, SERVING_DIR, load_params, setup_mlflow
from sentiment.gates import evaluate_gates


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def register() -> int:
    """Register the trained model. Only gate-passing models get the `candidate` alias."""
    params = load_params()
    name = params["registry"]["model_name"]
    run_id = json.loads((MODEL_DIR / "run_info.json").read_text())["run_id"]
    gates = json.loads((REPORT_DIR / "gates.json").read_text())
    metrics = json.loads((REPORT_DIR / "metrics.json").read_text())

    setup_mlflow()
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    tags = {
        "gates_passed": str(gates["passed"]),
        "test_macro_f1": f"{metrics['macro_f1']:.4f}",
        "git_commit": _git_sha(),
        "status": "candidate" if gates["passed"] else "rejected",
    }
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, v)
    if not gates["passed"]:
        failed = [c["name"] for c in gates["checks"] if not c["passed"]]
        client.set_model_version_tag(name, mv.version, "rejection_reason", ",".join(failed))
        print(f"Version {mv.version} REJECTED (failed: {failed})")
        return 1
    client.set_registered_model_alias(name, "candidate", mv.version)
    print(f"Version {mv.version} registered with alias 'candidate'")
    return 0


def export_champion() -> dict:
    """Write the champion model + metadata to SERVING_DIR for the inference API."""
    name = load_params()["registry"]["model_name"]
    setup_mlflow()
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, "champion")
    model = mlflow.sklearn.load_model(f"models:/{name}@champion")

    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    info = {
        "model_name": name,
        "version": mv.version,
        "run_id": mv.run_id,
        "test_macro_f1": mv.tags.get("test_macro_f1"),
        "approved_at": mv.tags.get("approved_at"),
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    joblib.dump(model, SERVING_DIR / "model.joblib")
    (SERVING_DIR / "serving.json").write_text(json.dumps(info, indent=2))
    return info


def promote_version(version: str | None = None) -> dict:
    """Approve a candidate as champion, re-checking gates against the current champion.

    Returns {"ok": bool, "version": str, "gates": {...}}; exports the model when ok.
    """
    params = load_params()
    name = params["registry"]["model_name"]
    setup_mlflow()
    client = MlflowClient()

    if version is None:
        version = client.get_model_version_by_alias(name, "candidate").version
    version = str(version)
    mv = client.get_model_version(name, version)
    run_metrics = client.get_run(mv.run_id).data.metrics
    metrics = {
        "accuracy": run_metrics["test_accuracy"],
        "macro_f1": run_metrics["test_macro_f1"],
        "min_class_recall": run_metrics["test_min_class_recall"],
    }
    try:
        champ = client.get_model_version_by_alias(name, "champion")
        # Re-promoting the current champion shouldn't be compared against itself.
        champ_f1 = (None if str(champ.version) == version
                    else client.get_run(champ.run_id).data.metrics.get("test_macro_f1"))
    except Exception:
        champ_f1 = None

    result = evaluate_gates(metrics, params["gates"], champ_f1)
    if not result.passed:
        return {"ok": False, "version": version, "gates": result.to_dict()}

    try:
        previous = str(client.get_model_version_by_alias(name, "champion").version)
    except Exception:
        previous = None
    client.set_registered_model_alias(name, "champion", version)
    if previous and previous != version:
        client.set_model_version_tag(name, previous, "status", "retired")
    client.set_model_version_tag(name, version, "status", "champion")
    client.set_model_version_tag(name, version, "approved_at",
                                 datetime.now(UTC).isoformat(timespec="seconds"))
    try:
        if str(client.get_model_version_by_alias(name, "candidate").version) == version:
            client.delete_registered_model_alias(name, "candidate")
    except Exception:
        pass
    export_champion()
    return {"ok": True, "version": version, "gates": result.to_dict()}


def promote(version: str | None) -> int:
    res = promote_version(version)
    if res["ok"]:
        print(f"Version {res['version']} is now 'champion' (exported to {SERVING_DIR})")
        return 0
    print("Promotion blocked:", json.dumps(res["gates"], indent=2))
    return 1


def status() -> int:
    name = load_params()["registry"]["model_name"]
    setup_mlflow()
    client = MlflowClient()
    alias_of: dict[str, list[str]] = {}
    for alias, ver in client.get_registered_model(name).aliases.items():
        alias_of.setdefault(str(ver), []).append(alias)
    for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda m: int(m.version)):
        t = mv.tags
        print(f"v{mv.version:<3} aliases={alias_of.get(str(mv.version), [])} "
              f"status={t.get('status')} f1={t.get('test_macro_f1')} commit={t.get('git_commit')}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["register", "promote", "export", "status"])
    ap.add_argument("--version")
    a = ap.parse_args()
    commands = {
        "register": register,
        "promote": lambda: promote(a.version),
        "export": lambda: print(json.dumps(export_champion(), indent=2)) or 0,
        "status": status,
    }
    sys.exit(commands[a.command]())
