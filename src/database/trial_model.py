"""SQLAlchemy ORM table for HP-tune trials."""

from __future__ import annotations

from typing import ClassVar

from sqlalchemy import Boolean, Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from .connection import Base


class TrialTable(Base):
    """Hyperparameter-tuning trial row."""

    __tablename__ = "trials"

    trial_id: Mapped[str] = mapped_column(Text, primary_key=True)
    lr: Mapped[float] = mapped_column(Float)
    epochs: Mapped[int] = mapped_column(Integer)
    dropout: Mapped[float] = mapped_column(Float)
    weight_decay: Mapped[float] = mapped_column(Float)
    batch_size: Mapped[int] = mapped_column(Integer)
    gradient_clip: Mapped[float] = mapped_column(Float)
    lr_scheduler: Mapped[bool] = mapped_column(Boolean, nullable=False)
    lr_scheduler_factor: Mapped[float] = mapped_column(Float)
    lr_scheduler_patience: Mapped[int] = mapped_column(Integer)
    early_stopping_patience: Mapped[int] = mapped_column(Integer)
    val_loss: Mapped[float] = mapped_column(Float)
    status: Mapped[int] = mapped_column(Integer, index=True)
    retries: Mapped[int] = mapped_column(Integer)

    def __repr__(self) -> str:
        return (
            f"TrialTable(trial_id={self.trial_id!r}, status={self.status!r}, "
            f"val_loss={self.val_loss!r})"
        )


TRIALS_TABLE = TrialTable.__tablename__
TRIAL_COLUMN_NAMES: ClassVar[tuple[str, ...]] = tuple(
    c.key for c in TrialTable.__table__.columns
)
