"""Service layer for CRUD operations on hyperparameter tuning trials (SQLite ORM)."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Sequence
from typing import Any, Mapping

import pandas as pd
from loguru import logger
from pydantic import TypeAdapter
from sqlalchemy import select, update

from database import Trial, get_db_session, engine
from schemas.trial_schema import HPTuneTrial, TrialStatus


class TrialService:
    """Static API for trial persistence and aggregates."""

    @staticmethod
    @logger.catch
    def get_trials() -> list[HPTuneTrial]:
        """Return all trials ordered by ``trial_id``."""
        with get_db_session() as session:
            rows = session.scalars(
                select(Trial).order_by(Trial.trial_id),
            ).all()
        logger.info(f"Fetched {len(rows)} trial row(s) from database.")
        return [HPTuneTrial.model_validate(r) for r in rows]

    @staticmethod
    @logger.catch
    def get_trial(trial_id: str) -> HPTuneTrial:
        """Return one trial by primary key."""
        with get_db_session() as session:
            row = session.get(Trial, trial_id)

        if row is None:
            raise ValueError(f"Trial with trial_id={trial_id} does not exist.")

        return HPTuneTrial.model_validate(row)

    @staticmethod
    @logger.catch
    def delete_trial(trial_id: str) -> bool:
        """Delete a trial by id. Returns ``True`` if a row was removed."""
        with get_db_session() as session:
            row = session.get(Trial, trial_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
        return True

    @staticmethod
    @logger.catch
    def update_trial(trial_id: str, payload: dict[str, Any]) -> None:
        """Partial update by primary key; ``status`` values persist as integers."""
        with get_db_session() as session:
            TypeAdapter(Mapping[str, Any]).validate_python(payload)
            values = {
                k: int(v) if k == "status" else v for k, v in payload.items()
            }
            session.execute(
                update(Trial).where(Trial.trial_id == trial_id).values(**values)
            )
            session.commit()

    @staticmethod
    @logger.catch
    def save_trials(trials: Sequence[HPTuneTrial]) -> None:
        """Upsert each trial row (merge by ``trial_id``)."""
        with get_db_session() as session:
            for trial in trials:
                session.merge(
                    Trial(
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
                        status=int(trial.status),
                        retries=trial.retries,
                    )
                )
            session.commit()
        logger.info(f"Upserted {len(trials)} trial(s).")

    @staticmethod
    @logger.catch
    def sql_to_csv() -> None:
        table_name = Trial.__table__.name
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        output_path = os.path.join(
            os.environ.get("TRIALS_DIR"), f"{table_name.lower()}.csv"
        )
        df.to_csv(output_path, index=False, encoding="utf-8")

    @staticmethod
    def get_status_counts(trials: list[HPTuneTrial]) -> dict[str, int]:
        """Aggregate counts by :class:`TrialStatus`."""
        by_status = Counter(t.status for t in trials)
        return {
            "done": by_status.get(TrialStatus.COMPLETED, 0),
            "running": by_status.get(TrialStatus.RUNNING, 0),
            "queued": by_status.get(TrialStatus.QUEUED, 0),
            "failed": by_status.get(TrialStatus.FAILED, 0),
            "total": len(trials),
        }
