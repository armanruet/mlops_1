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

## Roadmap
M2 DVC + MLflow + quality gates · M3 CI + Control Center · M4 monitoring/drift · M5 feedback + retraining
