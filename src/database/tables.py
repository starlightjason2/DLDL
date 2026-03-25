"""SQLAlchemy ORM table for HP-tune trials.

``trial_id`` is the sole primary key (natural key: ``trial_1``, ``trial_2``, …).
There is no surrogate ``id`` column. Expect a new database for each schema change.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .connection import engine


class Base(DeclarativeBase):
    pass


class Trial(Base):
    """One row per hyperparameter trial; keyed only by ``trial_id``."""

    __tablename__ = "trials"

    trial_id: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
        doc="Primary key (folder name under trials/).",
    )
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"TrialTable(trial_id={self.trial_id!r}, status={self.status!r}, "
            f"val_loss={self.val_loss!r})"
        )


Base.metadata.create_all(bind=engine, checkfirst=True)
