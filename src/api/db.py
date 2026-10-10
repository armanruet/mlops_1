"""PostgreSQL access: prediction log (and later reviewed feedback)."""

import logging
import os
from datetime import datetime

import pandas as pd
from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, func, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logger = logging.getLogger("api.db")

_engine: Engine | None = None


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


def get_engine() -> Engine | None:
    """Lazily create the engine. Returns None when no DATABASE_URL is configured."""
    global _engine
    if _engine is None and database_url():
        _engine = create_engine(database_url(), pool_pre_ping=True)
    return _engine


def log_prediction(review: str, label: str, confidence: float, model_version: str) -> bool:
    """Persist one prediction. Never raises: logging must not break serving."""
    try:
        engine = get_engine()
        if engine is None:
            return False
        with Session(engine) as session:
            session.add(Prediction(text=review, text_length=len(review), predicted_label=label,
                                   confidence=confidence, model_version=model_version))
            session.commit()
        return True
    except Exception:  # noqa: BLE001 - deliberately broad, see docstring
        logger.exception("Failed to log prediction")
        return False


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
