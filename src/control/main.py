"""MLOps Control Center API: aggregates MLflow, DVC lineage, gate reports and serving state."""

import json
import os
import secrets
import urllib.request
from pathlib import Path

import yaml
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow import MlflowClient
from pydantic import BaseModel

from sentiment.config import EXPERIMENT, PROCESSED_DIR, REPORT_DIR, load_params, setup_mlflow
from sentiment.registry import promote_version

API_URL = os.getenv("API_URL", "http://localhost:8000")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app = FastAPI(title="MLOps Control Center API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"],
                   allow_headers=["*"])


def _model_name() -> str:
    return load_params()["registry"]["model_name"]


def _client() -> MlflowClient:
    setup_mlflow()
    return MlflowClient()


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _http_json(url: str, method: str = "GET", headers: dict | None = None):
    req = urllib.request.Request(url, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.loads(r.read())


def _alias_map(client: MlflowClient, name: str) -> dict[str, list[str]]:
    """version -> aliases. search_model_versions doesn't populate aliases, the model does."""
    out: dict[str, list[str]] = {}
    for alias, version in client.get_registered_model(name).aliases.items():
        out.setdefault(str(version), []).append(alias)
    return out


def _version_dict(client: MlflowClient, mv, aliases: dict[str, list[str]]) -> dict:
    metrics = client.get_run(mv.run_id).data.metrics
    return {
        "version": mv.version,
        "run_id": mv.run_id,
        "aliases": aliases.get(str(mv.version), []),
        "tags": dict(mv.tags),
        "metrics": {k: v for k, v in metrics.items() if k.startswith("test_")},
    }


@app.get("/control/overview")
def overview():
    client, name = _client(), _model_name()
    try:
        serving = _http_json(f"{API_URL}/model-info")
    except Exception:
        serving = None
    versions = client.search_model_versions(f"name='{name}'")
    aliases = _alias_map(client, name)
    by_alias = {
        a: _version_dict(client, mv, aliases)
        for mv in versions
        for a in aliases.get(str(mv.version), [])
    }
    return {
        "serving": serving,
        "champion": by_alias.get("champion"),
        "candidate": by_alias.get("candidate"),
        "n_versions": len(versions),
        "latest_gates": _read_json(REPORT_DIR / "gates.json"),
        "latest_metrics": _read_json(REPORT_DIR / "metrics.json"),
        "gate_thresholds": load_params()["gates"],
    }


@app.get("/control/registry")
def registry():
    client, name = _client(), _model_name()
    aliases = _alias_map(client, name)
    versions = client.search_model_versions(f"name='{name}'")
    return sorted((_version_dict(client, mv, aliases) for mv in versions),
                  key=lambda v: int(v["version"]), reverse=True)


@app.get("/control/experiments")
def experiments(limit: int = 50):
    client = _client()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return []
    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=limit)
    return [
        {
            "run_id": r.info.run_id,
            "name": r.info.run_name,
            "status": r.info.status,
            "started_at": r.info.start_time,
            "params": r.data.params,
            "metrics": r.data.metrics,
        }
        for r in runs
    ]


@app.get("/control/lineage")
def lineage():
    """Pipeline stages with the exact data/code/param hashes recorded in dvc.lock."""
    lock = yaml.safe_load(Path("dvc.lock").read_text()) if Path("dvc.lock").exists() else {}
    stages = []
    for name, s in lock.get("stages", {}).items():
        stages.append({
            "name": name,
            "cmd": s.get("cmd"),
            "deps": [{"path": d["path"], "md5": d.get("md5"), "size": d.get("size")}
                     for d in s.get("deps", [])],
            "outs": [{"path": o["path"], "md5": o.get("md5"), "size": o.get("size")}
                     for o in s.get("outs", [])],
            "params": s.get("params", {}),
        })
    return {
        "dataset": {"source": "https://huggingface.co/datasets/stanfordnlp/imdb"},
        "data_stats": _read_json(PROCESSED_DIR / "stats.json"),
        "stages": stages,
    }


class PromoteRequest(BaseModel):
    version: str | None = None


@app.post("/control/promote")
def promote(req: PromoteRequest, x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        result = promote_version(req.version)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Cannot promote: {e}") from e
    if result["ok"]:
        try:
            _http_json(f"{API_URL}/admin/reload", "POST", {"X-Admin-Token": ADMIN_TOKEN})
            result["reloaded"] = True
        except Exception:
            result["reloaded"] = False
    return result
