# Sentiment MLOps

IMDB review sentiment classifier (TF-IDF + Logistic Regression) with a full MLOps lifecycle:
versioned data and pipeline (DVC), experiment tracking and model registry (MLflow), quality gates,
serving (FastAPI), monitoring (Prometheus, Grafana, Evidently) and a Control Center UI (Next.js).
Dataset: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb).

## Services

| Service | URL | Purpose |
|---|---|---|
| web | http://localhost:3000 | Classifier UI; Control Center at `/control` |
| api | http://localhost:8000 | `/predict`, `/health`, `/metrics`, `/model-info`, `/admin/reload` |
| control | http://localhost:8001 | Control Center API (registry, lineage, drift, promote) |
| mlflow | http://localhost:5001 | Tracking server + model registry (Postgres backend) |
| grafana | http://localhost:3001 | Dashboards (login `admin` / `admin`, change it) |
| prometheus | http://localhost:9090 | Metrics + alert rules |
| postgres | internal | Prediction log (`app` DB) and MLflow backend (`mlflow` DB) |

MLflow uses host port 5001 because macOS AirPlay Receiver occupies 5000.

## First-time setup

```bash
uv sync                                   # Python env (mlflow, dvc, evidently included)
docker compose up -d --build --wait postgres mlflow
uv run dvc repro                          # download -> prepare -> train -> evaluate -> register
uv run python -m sentiment.registry promote   # approve candidate -> champion, export for serving
docker compose up -d --build --wait       # everything else
```

Then open http://localhost:3000.

## Day to day

```bash
uv run dvc repro                          # retrain if data/params/code changed
uv run dvc push                           # push data/models to the DVC remote (./dvc-remote)
uv run python -m sentiment.registry status    # versions + aliases
uv run python -m sentiment.registry promote   # or use the Approve button in the Control Center
uv run python scripts/simulate_traffic.py --n 200            # normal traffic
uv run python scripts/simulate_traffic.py --n 300 --shift    # drifted traffic (demo)
uv run pytest && uv run ruff check src tests scripts migrations
cd web && npm install && npm run dev      # frontend dev server
```

## How it fits together

- **Pipeline** (`dvc.yaml`, params in `params.yaml`): download -> prepare -> train -> evaluate -> register.
- **Quality gates** (`src/sentiment/gates.py`): min accuracy, macro-F1, per-class recall, and no
  macro-F1 regression vs the current champion. Failing models are registered as `rejected` with a
  `rejection_reason` tag and the `register` stage exits non-zero.
- **Aliases**: `candidate` = passed gates, awaiting approval; `champion` = approved and serving.
  Promotion re-checks the gates, exports the model to `models/serving/` and tells the API to reload.
  Superseded champions are tagged `retired`.
- **Serving**: the API serves the exported champion, logs every prediction to Postgres (as a
  background task, so logging failures never break predictions) and exposes Prometheus metrics:
  request rate/latency/errors, predictions by label, confidence, input length, model version.
- **Drift** (`src/monitoring/drift.py`): Evidently compares the last `monitoring.window`
  predictions against a held-out reference scored by the serving model, on text statistics
  (length, word count, avg word length, exclamations, uppercase ratio) plus predicted label and
  confidence. Dataset drift = share of drifted columns >= `drift_share_threshold`. Runs every
  `DRIFT_INTERVAL_MIN` minutes (default 15), on demand from the Drift page, or via
  `python -m monitoring.drift`. Result feeds the `drift_detected` Prometheus gauge and an alert.
- **Migrations**: Alembic (`migrations/`); the API container runs `alembic upgrade head` on start.
- **CI** (`.github/workflows/ci.yml`): ruff, pytest, `dvc dag`, web lint/build, Docker image builds,
  compose validation.

## Configuration

Set via environment (or a `.env` file next to `docker-compose.yml`):
`ADMIN_TOKEN` (default `dev-admin-token`), `POSTGRES_PASSWORD`, `GRAFANA_PASSWORD`,
`DRIFT_INTERVAL_MIN`. The defaults are for local development only.

## Roadmap
M5 reviewed feedback + retraining (PostgreSQL, DVC, MLflow)
