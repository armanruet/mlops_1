"""Create predictions table.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_length", sa.Integer(), nullable=False),
        sa.Column("predicted_label", sa.String(16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_table("predictions")
