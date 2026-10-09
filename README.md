# Sentiment MLOps

IMDB review sentiment classifier (TF-IDF + Logistic Regression) with an MLOps lifecycle.
Dataset: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb).

## Quick start (M1 vertical slice)

```bash
uv sync
uv run python -m sentiment.data    # download + validate data
uv run python -m sentiment.train   # train, writes models/model.joblib
uv run pytest
uv run uvicorn api.main:app --reload   # API on :8000

cd web && npm install && npm run dev   # UI on :3000
```

Or with Docker (after training): `docker compose up --build`.

## Layout
- `src/sentiment/` data + training
- `src/api/` FastAPI service (`/health`, `/predict`)
- `web/` Next.js UI
- `tests/` pytest

## ML pipeline (M2)

```bash
uv run dvc repro                          # download -> prepare -> train -> evaluate -> register
uv run dvc push                           # push data/models to the DVC remote (./dvc-remote)
uv run python -m sentiment.registry status    # list versions + aliases
uv run python -m sentiment.registry promote   # approve `candidate` -> `champion` (re-checks gates)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse experiments/registry
```

- Params and quality-gate thresholds live in `params.yaml`.
- Gates (`src/sentiment/gates.py`): min accuracy, min macro-F1, min per-class recall, and no
  macro-F1 regression vs the current champion. Failing models are registered as `rejected`
  with a `rejection_reason` tag and the `register` stage exits non-zero.
- Aliases: `candidate` = passed gates, awaiting approval; `champion` = approved for serving.

## Serving + Control Center (M3)

Promotion exports the `champion` model to `models/serving/` (git-ignored), which the API serves.
The API reloads it on `POST /admin/reload` (needs `X-Admin-Token`), so no redeploy is needed.

```bash
docker compose up --build -d     # inference API :8000 + web :3000
ADMIN_TOKEN=dev-admin-token uv run uvicorn control.main:app --port 8001   # Control Center API
```

Open http://localhost:3000/control: Overview (serving vs champion, latest gates), Data & pipeline
(dvc.lock hashes), Experiments (MLflow runs), Model registry (approve candidate -> champion).
Approval requires the admin token (`ADMIN_TOKEN`, default `dev-admin-token` in compose; change it).
The Control Center API runs on the host for now because MLflow stores absolute artifact paths;
it moves into Compose with a proper MLflow server + Postgres in M4.

CI (`.github/workflows/ci.yml`): ruff, pytest, `dvc dag`, web lint/build, Docker image builds.

## Roadmap
M4 monitoring/drift (Prometheus, Grafana, Evidently) · M5 feedback + retraining
