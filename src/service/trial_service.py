"""Service layer for CRUD operations on hyperparameter tuning trials (SQLite ORM)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from database import (
    TRIAL_LOG_DB_FILENAME,
    PathLike,
    TrialTable,
    default_trial_db_path,
    initialize,
    trial_db_engine,
)
from model.hptune_trial import HPTuneTrial
from schemas.trial_schema import TrialSchema


class TrialService:
    """High-level API over the :mod:`database` trial log for single-trial CRUD and bulk snapshot."""

    #: Same basename as :data:`database.TRIAL_LOG_DB_FILENAME`.
    DEFAULT_DB_FILENAME = TRIAL_LOG_DB_FILENAME

    @classmethod
    def default_path(cls, trials_dir: str) -> str:
        """Resolve ``<trials_dir>/<DEFAULT_DB_FILENAME>`` for the trial log."""
        return default_trial_db_path(trials_dir)

    def __init__(self, db_path: PathLike) -> None:
        self._db_path = Path(db_path)
        initialize(self._db_path)

    @property
    def db_path(self) -> str:
        return str(self._db_path)

    def get_trials(self) -> list[HPTuneTrial]:
        """Return all trials ordered by ``trial_id``."""
        path = self._db_path
        os.makedirs(path.parent, exist_ok=True)
        initialize(path)
        with Session(trial_db_engine(path)) as session:
            rows = session.scalars(
                select(TrialTable).order_by(TrialTable.trial_id),
            ).all()
        return [HPTuneTrial(**TrialSchema.model_validate(r).model_dump()) for r in rows]

    def get_trial(self, trial_id: str) -> HPTuneTrial | None:
        """Return one trial by primary key, or ``None`` if missing."""
        with Session(trial_db_engine(self._db_path)) as session:
            row = session.get(TrialTable, trial_id)
        if row is None:
            return None
        return HPTuneTrial(**TrialSchema.model_validate(row).model_dump())

    def create_trial(self, trial: HPTuneTrial) -> HPTuneTrial:
        """Insert a new trial row. Raises ``ValueError`` if ``trial_id`` is missing or already exists."""
        trial.validate_for_persistence()
        with Session(trial_db_engine(self._db_path)) as session:
            if session.get(TrialTable, trial.trial_id) is not None:
                raise ValueError(f"Trial {trial.trial_id!r} already exists")
            session.add(
                TrialTable(
                    trial_id=trial.trial_id,
                    lr=trial.lr,
                    epochs=trial.epochs,
                    dropout=trial.dropout,
                    weight_decay=trial.weight_decay,
                    batch_size=trial.batch_size,
                    gradient_clip=trial.gradient_clip,
                    lr_scheduler=trial.lr_scheduler,
                    lr_scheduler_factor=trial.lr_scheduler_factor,
                    lr_scheduler_patience=trial.lr_scheduler_patience,
                    early_stopping_patience=trial.early_stopping_patience,
                    val_loss=trial.val_loss,
                    status=trial.status,
                    retries=trial.retries,
                )
            )
            session.commit()
        return trial

    def update_trial(self, trial_id: str, trial: HPTuneTrial) -> HPTuneTrial:
        """Update an existing row. ``trial`` is stored with ``trial_id`` forced to ``trial_id``."""
        merged = replace(trial, trial_id=trial_id)
        merged.validate_for_persistence()
        with Session(trial_db_engine(self._db_path)) as session:
            row = session.get(TrialTable, trial_id)
            if row is None:
                raise KeyError(f"No trial with trial_id={trial_id!r}")
            new_row = TrialTable(
                trial_id=merged.trial_id,
                lr=merged.lr,
                epochs=merged.epochs,
                dropout=merged.dropout,
                weight_decay=merged.weight_decay,
                batch_size=merged.batch_size,
                gradient_clip=merged.gradient_clip,
                lr_scheduler=merged.lr_scheduler,
                lr_scheduler_factor=merged.lr_scheduler_factor,
                lr_scheduler_patience=merged.lr_scheduler_patience,
                early_stopping_patience=merged.early_stopping_patience,
                val_loss=merged.val_loss,
                status=merged.status,
                retries=merged.retries,
            )
            for col in TrialTable.__table__.columns:
                if col.primary_key:
                    continue
                setattr(row, col.key, getattr(new_row, col.key))
            session.commit()
        return merged

    def delete_trial(self, trial_id: str) -> bool:
        """Delete a trial by id. Returns ``True`` if a row was removed."""
        with Session(trial_db_engine(self._db_path)) as session:
            row = session.get(TrialTable, trial_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def persist_snapshot(self, trials: Sequence[HPTuneTrial]) -> None:
        """Replace the entire ``trials`` table with the given snapshot (used by the tuner)."""
        path = self._db_path
        os.makedirs(path.parent, exist_ok=True)
        initialize(path)
        orm_rows: list[TrialTable] = []
        for t in trials:
            t.validate_for_persistence()
            orm_rows.append(
                TrialTable(
                    trial_id=t.trial_id,
                    lr=t.lr,
                    epochs=t.epochs,
                    dropout=t.dropout,
                    weight_decay=t.weight_decay,
                    batch_size=t.batch_size,
                    gradient_clip=t.gradient_clip,
                    lr_scheduler=t.lr_scheduler,
                    lr_scheduler_factor=t.lr_scheduler_factor,
                    lr_scheduler_patience=t.lr_scheduler_patience,
                    early_stopping_patience=t.early_stopping_patience,
                    val_loss=t.val_loss,
                    status=t.status,
                    retries=t.retries,
                )
            )
        with Session(trial_db_engine(path)) as session:
            session.execute(delete(TrialTable))
            if orm_rows:
                session.add_all(orm_rows)
            session.commit()
