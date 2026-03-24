"""Pydantic trial DTOs — API layer separate from :class:`database.trial_model.TrialTable` ORM."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TrialSchema(BaseModel):
    """Validated trial record for APIs and transport (not the SQLAlchemy table).

    Field types mirror :class:`~database.trial_model.TrialTable` (e.g. ``lr_scheduler`` is ``bool`` on both).

    ``from_attributes=True`` allows construction from ORM rows, e.g.
    ``TrialSchema.model_validate(trial_table_row)``.
    """

    model_config = ConfigDict(from_attributes=True)

    trial_id: str | None = None
    lr: float
    epochs: int
    dropout: float
    weight_decay: float
    batch_size: int
    gradient_clip: float
    lr_scheduler: bool
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    early_stopping_patience: int
    val_loss: float = -1.0
    status: int = -1
    retries: int = 0
