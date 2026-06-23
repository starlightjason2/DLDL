"""Service layer for CRUD operations on hyperparameter tuning trials (CSV)."""

from __future__ import annotations

import fcntl
import os
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd
from loguru import logger
from pydantic import TypeAdapter

from model.hp_trial import HPTuneTrial, TrialStatus

_CSV_COLUMNS = [
    "trial_id",
    "lr",
    "epochs",
    "dropout",
    "weight_decay",
    "batch_size",
    "gradient_clip",
    "lr_scheduler",
    "lr_scheduler_factor",
    "lr_scheduler_patience",
    "early_stopping_patience",
    "cls_pos_weight",
    "decision_threshold",
    "score",
    "status",
    "retries",
    "created_at",
    "updated_at",
]


class TrialService:
    """Static API for trial persistence and aggregates."""

    @staticmethod
    def _csv_path() -> Path:
        return Path(os.environ["HPTUNE_DIR"]) / "trials" / "trials.csv"

    @staticmethod
    @contextmanager
    def _locked_csv() -> Iterator[Path]:
        path = TrialService._csv_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".csv.lock")
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield path
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        return pd.DataFrame(columns=_CSV_COLUMNS)

    @staticmethod
    def _read_df(path: Path) -> pd.DataFrame:
        if not path.exists():
            return TrialService._empty_df()
        df = pd.read_csv(path, encoding="utf-8")
        for column in _CSV_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
        return df[_CSV_COLUMNS]

    @staticmethod
    def _write_df(df: pd.DataFrame, path: Path) -> None:
        tmp_path = path.with_suffix(".csv.tmp")
        df.sort_values("trial_id").to_csv(tmp_path, index=False, encoding="utf-8")
        tmp_path.replace(path)

    @staticmethod
    def _row_to_trial(row: pd.Series) -> HPTuneTrial:
        data = row.to_dict()
        for key in ("created_at", "updated_at"):
            if pd.isna(data.get(key)):
                data[key] = None
        if pd.isna(data.get("cls_pos_weight")):
            data["cls_pos_weight"] = 1.0
        if pd.isna(data.get("decision_threshold")):
            data["decision_threshold"] = 0.5
        if pd.isna(data.get("score")):
            data["score"] = -1.0
        data["status"] = int(data["status"])
        return HPTuneTrial.model_validate(data)

    @staticmethod
    def _trial_to_row(trial: HPTuneTrial) -> dict[str, Any]:
        return {
            "trial_id": trial.trial_id,
            "lr": trial.lr,
            "epochs": trial.epochs,
            "dropout": trial.dropout,
            "weight_decay": trial.weight_decay,
            "batch_size": trial.batch_size,
            "gradient_clip": trial.gradient_clip,
            "lr_scheduler": trial.lr_scheduler,
            "lr_scheduler_factor": trial.lr_scheduler_factor,
            "lr_scheduler_patience": trial.lr_scheduler_patience,
            "early_stopping_patience": trial.early_stopping_patience,
            "cls_pos_weight": trial.cls_pos_weight,
            "decision_threshold": trial.decision_threshold,
            "score": trial.score,
            "status": int(trial.status),
            "retries": trial.retries,
        }

    @staticmethod
    @logger.catch
    def get_trials() -> list[HPTuneTrial]:
        """Return all trials ordered by ``trial_id``."""
        with TrialService._locked_csv() as path:
            df = TrialService._read_df(path)
        trials = [TrialService._row_to_trial(row) for _, row in df.iterrows()]
        logger.info(f"Fetched {len(trials)} trial row(s) from CSV.")
        return trials

    @staticmethod
    @logger.catch
    def get_trial(trial_id: str) -> HPTuneTrial:
        """Return one trial by primary key."""
        with TrialService._locked_csv() as path:
            df = TrialService._read_df(path)
        matches = df[df["trial_id"] == trial_id]
        if matches.empty:
            raise ValueError(f"Trial with trial_id={trial_id} does not exist.")
        return TrialService._row_to_trial(matches.iloc[0])

    @staticmethod
    @logger.catch
    def update_trial(trial_id: str, payload: dict[str, Any]) -> None:
        """Partial update by primary key; ``status`` values persist as integers."""
        TypeAdapter(Mapping[str, Any]).validate_python(payload)
        values = {k: int(v) if k == "status" else v for k, v in payload.items()}

        with TrialService._locked_csv() as path:
            df = TrialService._read_df(path)
            matches = df.index[df["trial_id"] == trial_id]
            if matches.empty:
                raise ValueError(f"Trial with trial_id={trial_id} does not exist.")
            idx = matches[0]
            for key, value in values.items():
                df.at[idx, key] = value
            df.at[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
            TrialService._write_df(df, path)

    @staticmethod
    @logger.catch
    def save_trials(trials: Sequence[HPTuneTrial]) -> None:
        """Upsert each trial row (merge by ``trial_id``)."""
        now = datetime.now(timezone.utc).isoformat()

        with TrialService._locked_csv() as path:
            df = TrialService._read_df(path)
            for trial in trials:
                row = TrialService._trial_to_row(trial)
                matches = df.index[df["trial_id"] == trial.trial_id]
                if matches.empty:
                    row["created_at"] = now
                    row["updated_at"] = now
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                else:
                    idx = matches[0]
                    row["created_at"] = df.at[idx, "created_at"]
                    row["updated_at"] = now
                    for key, value in row.items():
                        df.at[idx, key] = value
            TrialService._write_df(df, path)

        logger.info(f"Upserted {len(trials)} trial(s).")

    @staticmethod
    def get_status_counts(trials: list[HPTuneTrial]) -> dict[str, int]:
        """Aggregate counts by :class:`TrialStatus`.

        ``active`` is running plus queued (work still in the pipeline); used by
        :meth:`model.bayesian_hptuner.BayesianHPTuner.run` dispatch logging.
        """
        by_status = Counter(t.status for t in trials)
        running = by_status.get(TrialStatus.RUNNING, 0)
        queued = by_status.get(TrialStatus.QUEUED, 0)
        return {
            "done": by_status.get(TrialStatus.COMPLETED, 0),
            "running": running,
            "queued": queued,
            "failed": by_status.get(TrialStatus.FAILED, 0),
            "total": len(trials),
            "active": running + queued,
        }
