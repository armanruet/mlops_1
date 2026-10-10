import pytest
from sqlalchemy import create_engine

from api import db
from api.db import (
    APPROVED,
    PENDING,
    REJECTED,
    Base,
    approved_feedback,
    feedback_stats,
    list_feedback,
    log_prediction,
    review_feedback,
    submit_feedback,
)


@pytest.fixture(autouse=True)
def sqlite_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "_engine", engine)
    yield engine
    Base.metadata.drop_all(engine)


def test_log_prediction_returns_id():
    pid = log_prediction("A fantastic film!", "positive", 0.95, "v1")
    assert pid is not None
    assert pid > 0


def test_submit_and_list_feedback():
    pid = log_prediction("A fantastic film!", "positive", 0.95, "v1")
    assert pid is not None

    fid = submit_feedback(pid, "negative")
    assert fid > 0

    pending = list_feedback(status=PENDING)
    assert len(pending) == 1
    assert pending[0]["id"] == fid
    assert pending[0]["user_label"] == "negative"
    assert pending[0]["predicted_label"] == "positive"
    assert pending[0]["review_status"] == PENDING


def test_submit_feedback_unknown_prediction():
    with pytest.raises(LookupError, match="Unknown prediction"):
        submit_feedback(9999, "positive")


def test_submit_feedback_invalid_label():
    pid = log_prediction("A fantastic film!", "positive", 0.95, "v1")
    assert pid is not None
    with pytest.raises(ValueError, match="label must be one of"):
        submit_feedback(pid, "neutral")


def test_review_feedback_approve_and_reject():
    pid1 = log_prediction("A fantastic film!", "positive", 0.95, "v1")
    pid2 = log_prediction("Worst movie ever", "negative", 0.92, "v1")
    assert pid1 is not None and pid2 is not None

    fid1 = submit_feedback(pid1, "negative")
    fid2 = submit_feedback(pid2, "positive")

    # Approve fid1 with its suggested user label
    rev1 = review_feedback(fid1, decision="approve", reviewer="tester")
    assert rev1["review_status"] == APPROVED
    assert rev1["reviewed_label"] == "negative"

    # Reject fid2
    rev2 = review_feedback(fid2, decision="reject", reviewer="tester")
    assert rev2["review_status"] == REJECTED
    assert rev2["reviewed_label"] is None

    # Check approved_feedback dataframe
    df = approved_feedback()
    assert len(df) == 1
    assert df.iloc[0]["feedback_id"] == fid1
    assert df.iloc[0]["label"] == 0  # negative maps to 0
    assert df.iloc[0]["text"] == "A fantastic film!"

    # Check stats
    stats = feedback_stats()
    assert stats["pending"] == 0
    assert stats["approved"] == 1
    assert stats["rejected"] == 1
    assert stats["error_rate_by_version"]["v1"]["reviewed"] == 1
    assert stats["error_rate_by_version"]["v1"]["wrong"] == 1
    assert stats["error_rate_by_version"]["v1"]["wrong_rate"] == 1.0


def test_review_feedback_with_relabel():
    pid = log_prediction("Neutral review", "positive", 0.60, "v1")
    assert pid is not None
    fid = submit_feedback(pid, "positive")

    # Human reviewer overrides label to negative
    rev = review_feedback(fid, decision="approve", label="negative", reviewer="admin")
    assert rev["review_status"] == APPROVED
    assert rev["reviewed_label"] == "negative"


def test_cannot_resubmit_reviewed_feedback():
    pid = log_prediction("Great film", "positive", 0.90, "v1")
    assert pid is not None
    fid = submit_feedback(pid, "positive")
    review_feedback(fid, decision="approve")

    with pytest.raises(ValueError, match="already reviewed"):
        submit_feedback(pid, "negative")
