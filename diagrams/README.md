# Sentiment MLOps Architectural & Engineering Diagrams

Interactive, showcase-quality diagrams created with [Archify](https://github.com/tt-a1i/archify) visualizing the complete Sentiment MLOps platform across all milestones (**M0 – M5**).

Each diagram is rendered as an interactive, standalone HTML application featuring inline SVG, dark/light theme switching, guided views, trace animation, relationship tracing, and clean vector export.

---

## Diagram Suite Overview

| Diagram | Specification | Delivered HTML | Purpose & Key Highlights |
|---|---|---|---|
| **System Architecture** | [`system-architecture.json`](./system-architecture.json) | [`system-architecture.html`](./system-architecture.html) | Multi-tier containerized architecture (`docker-compose`): Next.js web surface, FastAPI serving (`:8000`), Control Center API (`:8001`), PostgreSQL dual-database (`app` & `mlflow`), MLflow tracking (`:5001`), Prometheus (`:9090`), and Grafana (`:3001`). |
| **Continuous Retraining Workflow** | [`mlops-workflow.json`](./mlops-workflow.json) | [`mlops-workflow.html`](./mlops-workflow.html) | Human-in-the-loop retraining workflow (Schema v2): Feedback ingestion -> Human review queue -> Retraining trigger (`n >= 10`) -> DVC pipeline reproduction -> Quality gate evaluation -> MLflow `@champion` alias promotion -> Zero-downtime hot reload. |
| **Data Flow & Lineage** | [`mlops-dataflow.json`](./mlops-dataflow.json) | [`mlops-dataflow.html`](./mlops-dataflow.html) | End-to-end data pipelines across 5 stages: HuggingFace Stanford IMDB raw parquet -> Clean splits & reviewed feedback -> TF-IDF vectorizer + Logistic Regression -> Quality gate verification -> Serving inference audit log -> Evidently data drift surveillance. |
| **Model & Governance Lifecycle** | [`mlops-lifecycle.json`](./mlops-lifecycle.json) | [`mlops-lifecycle.html`](./mlops-lifecycle.html) | Complete milestone state machine (M0 through M5): Raw data intake -> Training run -> Gate evaluation -> Candidate model -> Champion hot serving -> Production drift monitoring -> Pending feedback -> Retraining trigger -> Model retirement. |
| **MLOps Toolchain Lifecycle Map** | [`mlops-tools-lifecycle.json`](./mlops-tools-lifecycle.json) | [`mlops-tools-lifecycle.html`](./mlops-tools-lifecycle.html) | Comprehensive toolchain lifecycle connecting all 12 tools (uv, GitHub Actions, DVC, Scikit-Learn, Quality Gates, MLflow, Docker Compose, FastAPI, Next.js, PostgreSQL/Alembic, Prometheus/Grafana, Evidently) across Foundation, Model/Serving, and Observability/Feedback phases. |

---

## Milestone Traceability (M0 – M5)

- **M0 (Problem & Environment)**: Raw dataset intake ([stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)), `uv` package environment, and Git version control. Visualized in the Data Flow Stage 0, Lifecycle Phase 01, and Toolchain Foundation.
- **M1 (Vertical Slice)**: Logistic Regression baseline (`ngram_max=2`, `C=2.0`), FastAPI serving endpoint (`POST /predict`), minimal web UI, and Docker Compose orchestration. Visualized in System Architecture, Workflow, and Toolchain Model/Serving.
- **M2 (DVC & MLflow)**: Deterministic DVC DAG (`download` -> `prepare` -> `export_feedback` -> `train` -> `evaluate` -> `register`), MLflow tracking server with PostgreSQL backend, automated quality gates (`macro_f1 >= 0.88`, `accuracy >= 0.88`, `recall >= 0.85`, max regression `<= 0.01`). Visualized in Workflow, Data Flow, and Toolchain Model/Serving.
- **M3 (Control Center & Hot Reload)**: Zero-downtime serving from the `@champion` registry alias, hot reload via `POST /admin/reload`, Control Center overview UI (`/control`), and GitHub Actions CI. Visualized in System Architecture, Workflow, and Toolchain Serving.
- **M4 (Monitoring & Drift)**: Prometheus metrics scraping (`/metrics`), Grafana dashboards (`:3001`), PostgreSQL prediction logging, and scheduled Evidently AI data drift detection (`DRIFT_INTERVAL_MIN`). Visualized in System Architecture, Data Flow, Lifecycle, and Toolchain Observability.
- **M5 (Human-in-the-Loop Feedback & Retraining)**: Feedback review queue (`POST /control/feedback/{id}/review`), anti-poisoning moderation, threshold retraining trigger (`min_reviewed: 10`), feedback parquet export, and champion promotion. Visualized in Workflow, Data Flow, Lifecycle, and Toolchain Feedback Loop.

---

## Viewing the Delivered Diagrams

Open any of the delivered HTML files directly in your web browser:
```bash
open diagrams/system-architecture.html
open diagrams/mlops-workflow.html
open diagrams/mlops-dataflow.html
open diagrams/mlops-lifecycle.html
open diagrams/mlops-tools-lifecycle.html
```

Or open the interactive portal:
```bash
open diagrams/index.html
```
