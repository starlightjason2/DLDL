"""Hyperparameter trial model (Pydantic), ORM bridge, and ``run.sh`` fragments."""

from __future__ import annotations

import os
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from util.hptune import write_env


class TrialStatus(IntEnum):
    """Persisted as integers in SQLite."""

    COMPLETED = 0
    RUNNING = -1
    QUEUED = -2
    FAILED = -3


TrialSignature = tuple[str, int, str, str, int, str, int, str, int, int]


class HPTuneTrial(BaseModel):
    """One hyperparameter trial: training hparams, log status, identity.

    ``trial_id`` is the primary key (same as :attr:`database.tables.Trial.trial_id`).
    """

    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    trial_id: str = Field(description="Primary key; ORM column ``trials.trial_id``.")
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
    status: TrialStatus = TrialStatus.RUNNING
    retries: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def dir_path(self) -> Path:
        """Absolute path to this trial's directory"""
        return Path(os.environ["HPTUNE_DIR"]) / "trials" / self.trial_id

    def signature(self) -> TrialSignature:
        """Normalize this trial into a hashable signature for duplicate detection."""
        return self.signature(self.model_dump())

    @classmethod
    def proposed_signature(cls, d: Mapping[str, Any]) -> TrialSignature:
        """Same tuple as :meth:`signature` for raw hyperparameter dicts."""
        return (
            f"{float(d['lr']):.12g}",
            int(d["epochs"]),
            f"{float(d['dropout']):.12g}",
            f"{float(d['weight_decay']):.12g}",
            int(d["batch_size"]),
            f"{float(d['gradient_clip']):.12g}",
            int(d["lr_scheduler"]),
            f"{float(d['lr_scheduler_factor']):.12g}",
            int(d["lr_scheduler_patience"]),
            int(d["early_stopping_patience"]),
        )

    def log_pass_hyperparameters(self, *, context: str) -> None:
        logger.opt(lazy=True).info(
            "Hyperparameters for this pass ({ctx}): trial_id={id} lr={lr:.2e} epochs={ep} "
            "dropout={do:.4f} weight_decay={wd:.2e} batch_size={bs} gradient_clip={gc:.3f} "
            "lr_scheduler={ls} lr_scheduler_factor={lf:.3f} lr_scheduler_patience={lp} "
            "early_stopping_patience={esp}",
            ctx=lambda: context,
            id=lambda: self.trial_id,
            lr=lambda: self.lr,
            ep=lambda: self.epochs,
            do=lambda: self.dropout,
            wd=lambda: self.weight_decay,
            bs=lambda: self.batch_size,
            gc=lambda: self.gradient_clip,
            ls=lambda: self.lr_scheduler,
            lf=lambda: self.lr_scheduler_factor,
            lp=lambda: self.lr_scheduler_patience,
            esp=lambda: self.early_stopping_patience,
        )

    def bayesian_params(self, batch_sizes: tuple[int, ...]) -> dict[str, float]:
        """Float-only parameter dict aligned with BayesianOptimization ``pbounds``."""
        batch_index = min(
            range(len(batch_sizes)),
            key=lambda i: abs(batch_sizes[i] - self.batch_size),
        )
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

    def trial_env_keys(self) -> dict[str, Any]:
        """Env vars written to ``<trial_dir>/.env`` — same mapping :meth:`create_scripts` persists."""
        return {
            "JOB_ID": self.trial_id,
            "LEARNING_RATE": str(self.lr),
            "NUM_EPOCHS": str(self.epochs),
            "BATCH_SIZE": str(self.batch_size),
            "DROPOUT_RATE": str(self.dropout),
            "WEIGHT_DECAY": str(self.weight_decay),
            "GRADIENT_CLIP": str(self.gradient_clip),
            "LR_SCHEDULER": str(self.lr_scheduler).lower(),
            "LR_SCHEDULER_FACTOR": str(self.lr_scheduler_factor),
            "LR_SCHEDULER_PATIENCE": str(self.lr_scheduler_patience),
            "EARLY_STOPPING_PATIENCE": str(self.early_stopping_patience),
            "PROG_DIR": self.dir_path,
        }

    @staticmethod
    def _build_run_script(
        project_root: str, env_path: Path, template_path: Path
    ) -> str:
        """Build ``run.sh`` from ``scripts/run_train.sh`` (``@...@`` placeholders)."""
        text = template_path.read_text()
        log_dir = Path(os.environ["HPTUNE_DIR"]) / "controller_logs"
        trial_boot = (
            f"cd {project_root}\n\n"
            "set -a\n"
            f"source {env_path}\n"
            "set +a\n"
            'export PROG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        )
        placeholders = {
            "@DLDL_ROOT@": project_root,
            "@HPTUNE_WALLTIME@": os.environ["HPTUNE_TRAIN_WALLTIME"],
            "@HPTUNE_QUEUE@": os.environ["HPTUNE_QUEUE"],
            "@HPTUNE_LOGDIR@": log_dir,
            "@DLDL_CD_AND_TRIAL_ENV@": trial_boot,
        }
        for key, val in placeholders.items():
            text = text.replace(key, str(val))
        leftover = [k for k in placeholders if k in text]
        if leftover:
            raise AssertionError(
                f"Unreplaced placeholders in {template_path}: {leftover}"
            )
        return text

    def create_scripts(
        self,
        *,
        project_root: str,
        env_lines: list[str] | None = None,
    ) -> str:
        """Write trial directory, ``.env``, and ``run.sh``; return ``trial_id``."""
        self.dir_path.mkdir(parents=True, exist_ok=True)
        env_path = self.dir_path / ".env"

        write_env(str(env_path), self.trial_env_keys(), env_lines)

        template_path = Path(project_root) / "scripts" / "run_train.sh"
        script = self._build_run_script(project_root, env_path, template_path)

        run_path = self.dir_path / "run.sh"
        run_path.write_text(script)
        run_path.chmod(0o755)

        return self.trial_id
