"""MLOps Control Center API: aggregates MLflow, DVC lineage, gate reports, drift and serving."""

import asyncio
import json
import logging
import os
import secrets
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from mlflow import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
from pydantic import BaseModel

from api.db import feedback_stats, list_feedback, review_feedback
from monitoring.drift import DRIFT_DIR, latest_summary, run_drift
from sentiment.config import EXPERIMENT, PROCESSED_DIR, REPORT_DIR, load_params, setup_mlflow
from sentiment.registry import promote_version
from sentiment.retrain import run_retraining

logger = logging.getLogger("control")

API_URL = os.getenv("API_URL", "http://localhost:8000")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
DRIFT_INTERVAL_MIN = float(os.getenv("DRIFT_INTERVAL_MIN", "0"))  # 0 disables the schedule

DRIFT_SHARE = Gauge("drift_share", "Share of monitored columns with detected drift")
DRIFT_DETECTED = Gauge("drift_detected", "1 if dataset drift detected in the latest report")
DRIFT_SAMPLES = Gauge("drift_current_samples", "Predictions in the latest drift window")


def _publish_drift(summary: dict | None) -> None:
    if not summary:
        return
    DRIFT_SAMPLES.set(summary.get("n_current", 0))
    if summary["status"] in ("ok", "drift"):
        DRIFT_SHARE.set(summary["drift_share"])
        DRIFT_DETECTED.set(1 if summary["status"] == "drift" else 0)


def _drift_job() -> dict:
    summary = run_drift()
    _publish_drift(summary)
    return summary


async def _drift_loop() -> None:
    while True:
        try:
            await asyncio.to_thread(_drift_job)
        except Exception:
            logger.exception("Scheduled drift run failed")
        await asyncio.sleep(DRIFT_INTERVAL_MIN * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _publish_drift(latest_summary())
    task = asyncio.create_task(_drift_loop()) if DRIFT_INTERVAL_MIN > 0 else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="MLOps Control Center API", version="0.2.0", lifespan=lifespan)
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
    try:
        fb_stats = feedback_stats()
        fb_stats["threshold"] = load_params().get("retraining", {}).get("min_reviewed", 10)
    except Exception:
        fb_stats = None
    return {
        "serving": serving,
        "champion": by_alias.get("champion"),
        "candidate": by_alias.get("candidate"),
        "n_versions": len(versions),
        "latest_gates": _read_json(REPORT_DIR / "gates.json"),
        "latest_metrics": _read_json(REPORT_DIR / "metrics.json"),
        "gate_thresholds": load_params()["gates"],
        "drift": latest_summary(),
        "feedback": fb_stats,
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


@app.get("/control/drift")
def drift():
    return latest_summary() or {"status": "no_report"}


@app.get("/control/drift/report", include_in_schema=False)
def drift_report():
    path = DRIFT_DIR / "latest.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No drift report yet")
    return FileResponse(path, media_type="text/html")


@app.post("/control/drift/run")
def drift_run(x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Forbidden")
    return _drift_job()


@app.get("/control/feedback")
def feedback_list(status: str = "pending", limit: int = 50):
    try:
        return list_feedback(status=status, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


class ReviewRequest(BaseModel):
    decision: str
    label: str | None = None


@app.post("/control/feedback/{feedback_id}/review")
def review(feedback_id: int, req: ReviewRequest, x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        return review_feedback(
            feedback_id, decision=req.decision, label=req.label, reviewer="admin"
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/control/feedback/stats")
def feedback_statistics():
    try:
        stats = feedback_stats()
        stats["threshold"] = load_params().get("retraining", {}).get("min_reviewed", 10)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


class RetrainRequest(BaseModel):
    force: bool = False
    min_reviewed: int | None = None


@app.post("/control/retrain")
async def retrain(req: RetrainRequest, x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        return await asyncio.to_thread(
            run_retraining, min_reviewed=req.min_reviewed, force=req.force
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}") from e


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

