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

_POST_TRAIN_FUNCS = r"""hptune_cleanup_epoch_checkpoints() {
  [[ -f "$PROG_DIR/${JOB_ID}_best_params.pt" ]] || return 0
  shopt -s nullglob
  rm -f "$PROG_DIR/${JOB_ID}_params_epoch"*.pt
  shopt -u nullglob
}

hptune_submit_next_serial_controller() {
  local c
  c=$(printf %s "$HPTUNE_CHAIN_ID" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  [[ -n "$c" ]] || return 0
  mkdir -p "${HPTUNE_DIR}/controller_logs"
  qsub -A "$HPTUNE_QSUB_ACCOUNT" -q "$HPTUNE_QUEUE" \
    -l "select=1:system=polaris,place=scatter,walltime=${HPTUNE_CONTROLLER_WALLTIME},filesystems=home:eagle" \
    -k doe -o "${HPTUNE_DIR}/controller_logs/" -e "${HPTUNE_DIR}/controller_logs/" \
    -v "HPTUNE_CHAIN_ID=$c" \
    "${PROJECT_ROOT}/scripts/controller.sh"
}
"""

_POST_TRAIN_RUN = '  [ "$RANK" = "0" ] && hptune_cleanup_epoch_checkpoints && hptune_submit_next_serial_controller'


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
        """Absolute path to this trial's directory under ``TRIALS_DIR``."""
        return Path(os.environ["TRIALS_DIR"]) / self.trial_id

    def trial_signature(self) -> TrialSignature:
        """Normalize this trial into a hashable signature for duplicate detection."""
        return self.signature_from_proposal(self.model_dump())

    @classmethod
    def signature_from_proposal(cls, d: Mapping[str, Any]) -> TrialSignature:
        """Same tuple as :meth:`trial_signature` for raw hyperparameter dicts."""
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

    @staticmethod
    def _build_run_script(project_root: str, env_path: str, template_path: str) -> str:
        """Build ``run.sh`` content from the shared training template."""
        template = Path(template_path).read_text()

        inject_block = (
            f"set -a\nsource {env_path}\nset +a\n"
            f'exec > >(tee "$PROG_DIR/train_${{PBS_JOBID}}.log") 2>&1'
        )

        hptune_dir = Path(os.environ["HPTUNE_DIR"])
        log_dir = hptune_dir / "controller_logs"
        walltime = os.environ["HPTUNE_TRAIN_WALLTIME"]
        pbs_queue = os.environ["HPTUNE_QUEUE"]

        script = (
            template.replace("#PBS -N dldl_train", "#PBS -N dldl_hptune")
            .replace("__DLDL_PROJECT_ROOT__", project_root)
            .replace("__HPTUNE_TRAIN_WALLTIME__", walltime)
            .replace("__HPTUNE_PBS_QUEUE__", pbs_queue)
            .replace("__HPTUNE_LOG_DIR__", str(log_dir))
            .replace("# __HPTUNE_CD_OVERRIDE__", f"cd {project_root}")
            .replace("# __HPTUNE_ENV_INJECT__", inject_block)
            .replace("# __HPTUNE_POST_TRAIN_FUNCS__\n", _POST_TRAIN_FUNCS + "\n")
            .replace("  # __HPTUNE_POST_TRAIN_RUN__", _POST_TRAIN_RUN)
        )

        sentinels = (
            "# __HPTUNE_CD_OVERRIDE__",
            "# __HPTUNE_ENV_INJECT__",
            "# __HPTUNE_POST_TRAIN_FUNCS__",
            "__HPTUNE_POST_TRAIN_RUN__",
            "__DLDL_PROJECT_ROOT__",
            "__HPTUNE_TRAIN_WALLTIME__",
            "__HPTUNE_PBS_QUEUE__",
            "__HPTUNE_LOG_DIR__",
        )
        for sentinel in sentinels:
            if sentinel in script:
                raise AssertionError(
                    f"Sentinel or placeholder not replaced in {template_path}: {sentinel!r}"
                )

        return script

    def create_scripts(
        self,
        *,
        project_root: str,
        env_lines: list[str] | None = None,
    ) -> str:
        """Write trial directory, ``.env``, and ``run.sh``; return ``trial_id``."""
        self.dir_path.mkdir(parents=True, exist_ok=True)
        env_path = self.dir_path / ".env"

        write_env(
            str(env_path),
            {
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
            },
            env_lines,
        )

        template_path = Path(project_root) / "scripts" / "run_train.sh"
        script = self._build_run_script(project_root, str(env_path), str(template_path))

        run_path = self.dir_path / "run.sh"
        run_path.write_text(script)
        run_path.chmod(0o755)

        return self.trial_id
