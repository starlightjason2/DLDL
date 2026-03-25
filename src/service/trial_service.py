"""Service layer for CRUD operations on hyperparameter tuning trials (SQLite ORM)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from loguru import logger
import pandas as pd
from sqlalchemy import delete, select
from database import Trial, get_db_session, engine
from schemas.trial_schema import HPTuneTrial


def get_trials() -> list[HPTuneTrial]:
    """Return all trials ordered by ``trial_id``."""
    with get_db_session() as session:
        rows = session.scalars(
            select(Trial).order_by(Trial.trial_id),
        ).all()
    logger.info(f"Fetched {len(rows)} trial row(s) from database.")
    return [HPTuneTrial.model_validate(r) for r in rows]


def get_trial(trial_id: str) -> HPTuneTrial | None:
    """Return one trial by primary key, or ``None`` if missing."""
    with get_db_session() as session:
        row = session.get(Trial, trial_id)
    if row is None:
        return None
    return HPTuneTrial.model_validate(row)


def update_trial(trial_id: str, trial: HPTuneTrial) -> HPTuneTrial:
    """Update an existing row by primary key ``trial_id``."""
    merged = trial.model_copy(update={"trial_id": trial_id})
    with get_db_session() as session:
        row = session.get(Trial, trial_id)
        if row is None:
            raise KeyError(f"No trial with trial_id={trial_id!r}")
        new_row = Trial(
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
        for col in Trial.__table__.columns:
            if not col.primary_key:
                setattr(row, col.key, getattr(new_row, col.key))

        session.commit()
    return merged


def delete_trial(trial_id: str) -> bool:
    """Delete a trial by id. Returns ``True`` if a row was removed."""
    with get_db_session() as session:
        row = session.get(Trial, trial_id)
        if row is None:
            return False
        session.delete(row)
        session.commit()
    return True


def save_trials(trials: Sequence[HPTuneTrial]) -> None:
    """Replace the entire ``trials`` table with the given snapshot (used by the tuner)."""
    with get_db_session() as session:
        orm_rows: list[Trial] = []
        for t in trials:
            orm_rows.append(
                Trial(
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

        if orm_rows:
            session.add_all(orm_rows)

        session.commit()
        logger.info(f"Appended {len(trials)} queued trial row(s) to database.")


def sql_to_csv():
    table_name = Trial.__table__.name
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    output_path = os.path.join(
        os.environ.get("TRIALS_DIR"), f"{table_name.lower()}.csv"
    )
    df.to_csv(output_path, index=False, encoding="utf-8")
