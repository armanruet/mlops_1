# Sentiment MLOps

IMDB review sentiment classifier (TF-IDF + Logistic Regression) with a full MLOps lifecycle:
versioned data and pipeline (DVC), experiment tracking and model registry (MLflow), quality gates,
serving (FastAPI), monitoring (Prometheus, Grafana, Evidently), human-in-the-loop reviewed feedback
and continuous retraining (PostgreSQL, MLflow, DVC), and a Control Center UI (Next.js).
Dataset: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb).

## Services

| Service | URL | Purpose |
|---|---|---|
| web | http://localhost:3000 | Classifier UI; Control Center at `/control` |
| api | http://localhost:8000 | `/predict`, `/feedback`, `/health`, `/metrics`, `/model-info`, `/admin/reload` |
| control | http://localhost:8001 | Control Center API (registry, lineage, drift, feedback, retrain, promote) |
| mlflow | http://localhost:5001 | Tracking server + model registry (Postgres backend) |
| grafana | http://localhost:3001 | Dashboards (login `admin` / `admin`) |
| prometheus | http://localhost:9090 | Metrics + alert rules |
| postgres | `localhost:5433` (internal `5432`) | Prediction log & feedback (`app` DB) and MLflow backend (`mlflow` DB) |

MLflow uses host port 5001 because macOS AirPlay Receiver occupies 5000. Postgres is mapped to host port 5433 to avoid conflicts if a local PostgreSQL service is running on 5432.

## First-time setup

```bash
uv sync                                   # Python env (mlflow, dvc, evidently included)
docker compose up -d --build --wait postgres mlflow
uv run dvc repro                          # download -> prepare -> train -> evaluate -> register
uv run python -m sentiment.registry promote   # approve candidate -> champion, export for serving
docker compose up -d --build --wait       # start all services
```

Then open http://localhost:3000 for the classifier, or http://localhost:3000/control for the Control Center.

## Day to day

```bash
uv run dvc repro                          # retrain via DVC if data/params/code changed
uv run dvc push                           # push data/models to the DVC remote (./dvc-remote)
uv run python -m sentiment.registry status    # view registered model versions + aliases
uv run python -m sentiment.registry promote   # or use the Approve button in the Control Center

# Traffic & feedback simulation
uv run python scripts/simulate_traffic.py --n 200            # normal traffic
uv run python scripts/simulate_traffic.py --n 300 --shift    # drifted traffic (triggers alert)
uv run python scripts/simulate_feedback.py --n 15            # simulate user corrections + review

# Verification & Dev
uv run pytest && uv run ruff check src tests scripts migrations
cd web && npm install && npm run dev      # frontend dev server
```

## How it fits together

- **Pipeline** (`dvc.yaml`, params in `params.yaml`): download -> prepare -> export_feedback -> train -> evaluate -> register.
- **Quality gates** (`src/sentiment/gates.py`): min accuracy (0.88), macro-F1 (0.88), per-class recall (0.85), and no macro-F1 regression vs the current champion (0.01 tolerance). Failing models are registered as `rejected` with a `rejection_reason` tag.
- **Aliases**: `candidate` = passed gates, awaiting human approval; `champion` = approved and serving. Promotion re-checks gates, exports the model to `models/serving/` and reloads the serving API with zero downtime. Superseded champions are tagged `retired`.
- **Serving & Prediction Logging**: the API serves the exported champion, returns prediction IDs, logs predictions to Postgres, and exposes Prometheus metrics: request rate/latency/errors, predictions by label, confidence histograms, input length, and active `model_info` gauge.
- **Human-in-the-Loop Feedback & Retraining** (`src/api/db.py`, `src/sentiment/retrain.py`, `/control/feedback`):
  1. Users submit suggested corrections from the web app (`POST /feedback`).
  2. Submissions enter a pending queue in Postgres, preventing untrusted labels from poisoning retraining.
  3. Reviewers approve, reject, or relabel items in the Control Center (`/control/feedback`).
  4. Once approved items meet the configured threshold (`retraining.min_reviewed: 10`), retraining can be triggered from the UI or CLI (`python -m sentiment.retrain run`).
  5. Approved feedback is exported to `data/feedback/reviewed.parquet` and merged with the base training set.
  6. The candidate model is trained, evaluated on held-out test data, tested against quality gates, and registered in MLflow.
  7. Human operators review candidate metrics and approve promotion to champion with one click.
- **Drift Detection** (`src/monitoring/drift.py`): Evidently compares recent predictions against a held-out reference on text features (length, word count, avg word length, exclamations, uppercase ratio) plus label distribution and confidence. Runs on a schedule (`DRIFT_INTERVAL_MIN`), on demand from `/control/drift`, and feeds Prometheus metrics + alerts.
- **Migrations**: Alembic (`migrations/`); `0001_create_predictions.py` and `0002_create_feedback.py`. Applied automatically on API startup.
- **CI** (`.github/workflows/ci.yml`): ruff, pytest, `dvc dag`, web lint/build, Docker image builds, and compose validation.

## Configuration

Set via environment (or a `.env` file next to `docker-compose.yml`):
`ADMIN_TOKEN` (default `dev-admin-token`), `POSTGRES_PASSWORD` (default `mlops-dev-password`), `GRAFANA_PASSWORD` (default `admin`), `DRIFT_INTERVAL_MIN` (default `15`).

## Lifecycle Milestones

- [x] **M1**: Vertical slice (TF-IDF + Logistic Regression, FastAPI inference, basic web UI).
- [x] **M2**: DVC pipeline, MLflow tracking, quality gates, and model registry aliases.
- [x] **M3**: Serving from champion alias, zero-downtime hot reload, CI workflow, Control Center overview.
- [x] **M4**: Prometheus & Grafana metrics/alerting, PostgreSQL prediction logging, Evidently data drift reports.
- [x] **M5**: Human-in-the-loop reviewed feedback, database migrations, continuous retraining pipeline, and candidate promotion.
