"""Create feedback table (user suggestions + human review).

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("prediction_id", sa.Integer(), sa.ForeignKey("predictions.id"), nullable=False,
                  unique=True),
        sa.Column("user_label", sa.String(16), nullable=True),
        sa.Column("reviewed_label", sa.String(16), nullable=True),
        sa.Column("review_status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reviewer", sa.String(64), nullable=True),
    )
    op.create_index("ix_feedback_review_status", "feedback", ["review_status"])


def downgrade() -> None:
    op.drop_index("ix_feedback_review_status", table_name="feedback")
    op.drop_table("feedback")
