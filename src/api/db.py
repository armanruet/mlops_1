"""PostgreSQL access: prediction log and reviewed feedback."""

import logging
import os
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logger = logging.getLogger("api.db")

_engine: Engine | None = None

PENDING, APPROVED, REJECTED = "pending", "approved", "rejected"
LABEL_IDS = {"negative": 0, "positive": 1}


def database_url() -> str:
    return os.getenv("DATABASE_URL", "")


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    text_length: Mapped[int] = mapped_column(Integer)
    predicted_label: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(32))


class Feedback(Base):
    """A user's suggested correction, plus the human reviewer's decision.

    Only `reviewed_label` of `approved` rows is ever used for training; `user_label` is a
    suggestion and is never trusted on its own.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("predictions.id"), unique=True
    )
    user_label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reviewed_label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    review_status: Mapped[str] = mapped_column(String(16), default=PENDING, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(64), nullable=True)


def get_engine() -> Engine | None:
    """Lazily create the engine. Returns None when no DATABASE_URL is configured."""
    global _engine
    if _engine is None and database_url():
        _engine = create_engine(database_url(), pool_pre_ping=True)
    return _engine


def _require_engine() -> Engine:
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Database is not configured")
    return engine


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def log_prediction(review: str, label: str, confidence: float, model_version: str) -> int | None:
    """Persist one prediction and return its id. Never raises: logging must not break serving."""
    try:
        engine = get_engine()
        if engine is None:
            return None
        with Session(engine) as session:
            row = Prediction(text=review, text_length=len(review), predicted_label=label,
                             confidence=confidence, model_version=model_version)
            session.add(row)
            session.commit()
            return row.id
    except Exception:  # noqa: BLE001 - deliberately broad, see docstring
        logger.exception("Failed to log prediction")
        return None


def recent_predictions(limit: int = 5000) -> pd.DataFrame:
    """Most recent predictions, newest first. Empty frame if the DB isn't configured."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    query = text(
        "SELECT id, created_at, text, predicted_label, confidence, model_version "
        "FROM predictions ORDER BY created_at DESC LIMIT :limit"
    )
    return pd.read_sql(query, engine, params={"limit": limit})


# --- feedback -----------------------------------------------------------------


def submit_feedback(prediction_id: int, user_label: str) -> int:
    """Record a user's suggested label for a prediction. Returns the feedback id.

    Raises LookupError (unknown prediction), ValueError (bad label / already reviewed),
    RuntimeError (no database).
    """
    if user_label not in LABEL_IDS:
        raise ValueError(f"label must be one of {sorted(LABEL_IDS)}")
    with Session(_require_engine()) as session:
        if session.get(Prediction, prediction_id) is None:
            raise LookupError("Unknown prediction")
        fb = session.scalar(select(Feedback).where(Feedback.prediction_id == prediction_id))
        if fb is None:
            fb = Feedback(prediction_id=prediction_id, user_label=user_label,
                          review_status=PENDING)
            session.add(fb)
        elif fb.review_status != PENDING:
            raise ValueError("Feedback for this prediction was already reviewed")
        else:
            fb.user_label = user_label
        session.commit()
        return fb.id


def _feedback_row(fb: Feedback, pred: Prediction) -> dict:
    return {
        "id": fb.id,
        "prediction_id": pred.id,
        "text": pred.text,
        "predicted_label": pred.predicted_label,
        "confidence": pred.confidence,
        "model_version": pred.model_version,
        "user_label": fb.user_label,
        "reviewed_label": fb.reviewed_label,
        "review_status": fb.review_status,
        "created_at": fb.created_at.isoformat() if fb.created_at else None,
        "reviewed_at": fb.reviewed_at.isoformat() if fb.reviewed_at else None,
    }


def list_feedback(status: str = PENDING, limit: int = 50) -> list[dict]:
    with Session(_require_engine()) as session:
        rows = session.execute(
            select(Feedback, Prediction)
            .join(Prediction, Prediction.id == Feedback.prediction_id)
            .where(Feedback.review_status == status)
            .order_by(Feedback.id.desc())
            .limit(limit)
        ).all()
        return [_feedback_row(fb, pred) for fb, pred in rows]


def review_feedback(feedback_id: int, decision: str, label: str | None = None,
                    reviewer: str = "admin") -> dict:
    """Approve (optionally with a corrected label) or reject a feedback item.

    Raises LookupError if missing, ValueError for invalid input.
    """
    if decision not in ("approve", "reject"):
        raise ValueError("decision must be 'approve' or 'reject'")
    with Session(_require_engine()) as session:
        fb = session.get(Feedback, feedback_id)
        if fb is None:
            raise LookupError("Unknown feedback")
        if decision == "approve":
            final = label or fb.user_label
            if final not in LABEL_IDS:
                raise ValueError(f"label must be one of {sorted(LABEL_IDS)}")
            fb.reviewed_label, fb.review_status = final, APPROVED
        else:
            fb.reviewed_label, fb.review_status = None, REJECTED
        fb.reviewed_at, fb.reviewer = datetime.now(UTC), reviewer
        session.commit()
        pred = session.get(Prediction, fb.prediction_id)
        return _feedback_row(fb, pred)


def approved_feedback() -> pd.DataFrame:
    """Approved reviews as training rows: text, label (0/1), plus provenance columns."""
    cols = ["feedback_id", "text", "label", "predicted_label", "model_version", "reviewed_at"]
    with Session(_require_engine()) as session:
        rows = session.execute(
            select(Feedback, Prediction)
            .join(Prediction, Prediction.id == Feedback.prediction_id)
            .where(Feedback.review_status == APPROVED)
            .order_by(Feedback.id)
        ).all()
    records = [
        {"feedback_id": fb.id, "text": p.text, "label": LABEL_IDS[fb.reviewed_label],
         "predicted_label": p.predicted_label, "model_version": p.model_version,
         "reviewed_at": fb.reviewed_at}
        for fb, p in rows
    ]
    return pd.DataFrame(records, columns=cols)


def feedback_stats(since: datetime | None = None) -> dict:
    """Review queue counts, served model error rate, and approvals since `since`."""
    with Session(_require_engine()) as session:
        counts = dict(session.execute(
            select(Feedback.review_status, func.count()).group_by(Feedback.review_status)
        ).all())
        approved = session.execute(
            select(Feedback, Prediction)
            .join(Prediction, Prediction.id == Feedback.prediction_id)
            .where(Feedback.review_status == APPROVED)
        ).all()
    errors: dict[str, dict[str, int]] = {}
    new_since = 0
    for fb, pred in approved:
        e = errors.setdefault(pred.model_version, {"reviewed": 0, "wrong": 0})
        e["reviewed"] += 1
        e["wrong"] += int(fb.reviewed_label != pred.predicted_label)
        if since is None or (fb.reviewed_at and _utc(fb.reviewed_at) > since):
            new_since += 1
    return {
        "pending": counts.get(PENDING, 0),
        "approved": counts.get(APPROVED, 0),
        "rejected": counts.get(REJECTED, 0),
        "approved_since_last_retrain": new_since,
        "error_rate_by_version": {
            v: {**e, "wrong_rate": e["wrong"] / e["reviewed"]} for v, e in errors.items()
        },
    }
