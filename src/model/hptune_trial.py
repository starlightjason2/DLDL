"""Trial model for DLDL Bayesian hyperparameter tuning."""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class HPTuneTrial:
    """One hyperparameter trial: non-architecture training hparams, log status, identity."""

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
    trial_id: Optional[str] = None
    val_loss: float = -1.0
    status: int = -1

    @property
    def dir_name(self) -> str:
        """Folder under ``trials/`` (``trial_1``, ``trial_2``, ...). Requires ``trial_id``."""
        if not self.trial_id:
            raise ValueError("trial_id must be set before using dir_name or path_under")
        return self.trial_id

    def path_under(self, trials_dir: str) -> str:
        return os.path.join(trials_dir, self.dir_name)

    @classmethod
    def from_series(cls, row: pd.Series) -> "HPTuneTrial":
        lr_scheduler_raw = row["lr_scheduler"]
        lr_scheduler = (
            bool(int(lr_scheduler_raw)) if not pd.isna(lr_scheduler_raw) else True
        )
        raw_trial_id = row["trial_id"]
        if raw_trial_id is None or (
            isinstance(raw_trial_id, float) and pd.isna(raw_trial_id)
        ):
            raise ValueError("trials_log.csv row is missing trial_id (required)")
        trial_id = str(raw_trial_id).strip()
        if not trial_id or trial_id.lower() in ("nan", "none"):
            raise ValueError("trials_log.csv row has empty trial_id (required)")
        return cls(
            lr=float(row["lr"]),
            epochs=int(row["epochs"]),
            dropout=float(row["dropout"]),
            weight_decay=float(row["weight_decay"]),
            batch_size=int(row["batch_size"]),
            gradient_clip=float(row["gradient_clip"]),
            lr_scheduler=lr_scheduler,
            lr_scheduler_factor=float(row["lr_scheduler_factor"]),
            lr_scheduler_patience=int(row["lr_scheduler_patience"]),
            early_stopping_patience=int(row["early_stopping_patience"]),
            trial_id=trial_id,
            val_loss=float(row["val_loss"]),
            status=int(row["status"]),
        )

    def to_csv_row(self) -> dict[str, float | int | str]:
        if not self.trial_id:
            raise ValueError("trial_id must be set before serializing to CSV")
        return {
            "trial_id": self.trial_id,
            "lr": self.lr,
            "epochs": self.epochs,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "gradient_clip": self.gradient_clip,
            "lr_scheduler": int(self.lr_scheduler),
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "early_stopping_patience": self.early_stopping_patience,
            "val_loss": self.val_loss,
            "status": self.status,
        }

    def bayesian_params(self, batch_sizes: tuple[int, ...]) -> dict[str, float]:
        """Float-only parameter dict aligned with BayesianOptimization pbounds."""
        batch_index = self._batch_index(batch_sizes)
        return {
            "lr": self.lr,
            "dropout": self.dropout,
            "log_wd": float(np.log10(max(self.weight_decay, 1e-20))),
            "epochs": float(self.epochs),
            "gradient_clip": self.gradient_clip,
            "lr_scheduler_u": 1.0 if self.lr_scheduler else 0.0,
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "lr_sched_patience": float(self.lr_scheduler_patience),
            "early_stop_patience": float(self.early_stopping_patience),
            "batch_idx": float(batch_index),
        }

    def write_env_file(self, env_path: str, env_lines: list[str]) -> None:
        """Write this trial's `.env` file from the shared template plus overrides."""
        lr_scheduler_str = "true" if self.lr_scheduler else "false"
        env_content = (
            "\n".join(env_lines)
            + f"""
    # HPTune overrides
    LEARNING_RATE={self.lr}
    NUM_EPOCHS={self.epochs}
    DROPOUT_RATE={self.dropout}
    WEIGHT_DECAY={self.weight_decay}
    BATCH_SIZE={self.batch_size}
    GRADIENT_CLIP={self.gradient_clip}
    LR_SCHEDULER={lr_scheduler_str}
    LR_SCHEDULER_FACTOR={self.lr_scheduler_factor}
    LR_SCHEDULER_PATIENCE={self.lr_scheduler_patience}
    EARLY_STOPPING_PATIENCE={self.early_stopping_patience}
    PROG_DIR={os.path.dirname(env_path)}
    JOB_ID={self.trial_id}
    # run.sh tee already writes full stderr to train_${{PBS_JOBID}}.log; skip duplicate training.log
    TRAIN_LOGURU_FILE=0
    """
        )
        with open(env_path, "w") as f:
            f.write(env_content)

    def _batch_index(self, batch_sizes: tuple[int, ...]) -> int:
        if self.batch_size in batch_sizes:
            return batch_sizes.index(self.batch_size)
        return min(
            range(len(batch_sizes)), key=lambda index: abs(batch_sizes[index] - self.batch_size)
        )
