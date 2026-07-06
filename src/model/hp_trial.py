"""Hyperparameter trial model (Pydantic), ORM bridge, and ``run.sh`` fragments."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from model.hyperparam_space import ArchitectureHyperparameterSpace, HyperparameterSpace
from model.trial_status import TrialStatus
from util.hptune import fixed_training_env_keys, write_env

TrainingTrialSignature = tuple[str, int, str, str, int, str, int, str, int, int, str]
ArchitectureTrialSignature = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]


class HPTuneTrial(BaseModel):
    """One HPTune trial: training hparams, optional architecture hparams, and status."""

    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    trial_id: str = Field(
        description="Primary key; column ``trial_id`` in ``trials.csv``."
    )
    job_id: str = ""
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
    cls_pos_weight: float = 1.0
    conv1_filters: int | None = None
    conv1_kernel: int | None = None
    conv1_padding: int | None = None
    conv2_filters: int | None = None
    conv2_kernel: int | None = None
    conv2_padding: int | None = None
    conv3_filters: int | None = None
    conv3_kernel: int | None = None
    conv3_padding: int | None = None
    conv4_filters: int | None = None
    conv4_kernel: int | None = None
    conv4_padding: int | None = None
    pool_size: int | None = None
    fc1_size: int | None = None
    fc2_size: int | None = None
    score: float = -1.0
    recall: float = -1.0
    precision: float = -1.0
    status: TrialStatus = TrialStatus.RUNNING
    retries: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def dir_path(self) -> Path:
        return Path(os.environ["HPTUNE_DIR"]) / "trials" / self.trial_id

    @property
    def is_architecture_trial(self) -> bool:
        return self.conv1_filters is not None

    def signature(self) -> TrainingTrialSignature | ArchitectureTrialSignature:
        return self.proposed_signature(self.model_dump())

    @classmethod
    def proposed_signature(
        cls, d: Mapping[str, Any]
    ) -> TrainingTrialSignature | ArchitectureTrialSignature:
        if d.get("conv1_filters") is not None:
            return (
                int(d["conv1_filters"]),
                int(d["conv1_kernel"]),
                int(d["conv1_padding"]),
                int(d["conv2_filters"]),
                int(d["conv2_kernel"]),
                int(d["conv2_padding"]),
                int(d["conv3_filters"]),
                int(d["conv3_kernel"]),
                int(d["conv3_padding"]),
                int(d["conv4_filters"]),
                int(d["conv4_kernel"]),
                int(d["conv4_padding"]),
                int(d["pool_size"]),
                int(d["fc1_size"]),
                int(d["fc2_size"]),
            )
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
            f"{float(d['cls_pos_weight']):.12g}",
        )

    def log_pass_hyperparameters(self, *, context: str) -> None:
        if self.is_architecture_trial:
            logger.opt(lazy=True).info(
                "Architecture trial ({ctx}): trial_id={id} "
                "conv1=({c1f},{c1k},{c1p}) conv2=({c2f},{c2k},{c2p}) "
                "conv3=({c3f},{c3k},{c3p}) conv4=({c4f},{c4k},{c4p}) "
                "pool={pool} fc1={fc1} fc2={fc2}",
                ctx=lambda: context,
                id=lambda: self.trial_id,
                c1f=lambda: self.conv1_filters,
                c1k=lambda: self.conv1_kernel,
                c1p=lambda: self.conv1_padding,
                c2f=lambda: self.conv2_filters,
                c2k=lambda: self.conv2_kernel,
                c2p=lambda: self.conv2_padding,
                c3f=lambda: self.conv3_filters,
                c3k=lambda: self.conv3_kernel,
                c3p=lambda: self.conv3_padding,
                c4f=lambda: self.conv4_filters,
                c4k=lambda: self.conv4_kernel,
                c4p=lambda: self.conv4_padding,
                pool=lambda: self.pool_size,
                fc1=lambda: self.fc1_size,
                fc2=lambda: self.fc2_size,
            )
            return

        logger.opt(lazy=True).info(
            "Hyperparameters for this pass ({ctx}): trial_id={id} lr={lr:.2e} epochs={ep} "
            "dropout={do:.4f} weight_decay={wd:.2e} batch_size={bs} gradient_clip={gc:.3f} "
            "lr_scheduler={ls} lr_scheduler_factor={lf:.3f} lr_scheduler_patience={lp} "
            "early_stopping_patience={esp} cls_pos_weight={cpw:.3f}",
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
            cpw=lambda: self.cls_pos_weight,
        )

    def bayesian_params(
        self, hp_space: HyperparameterSpace | ArchitectureHyperparameterSpace
    ) -> dict[str, float]:
        if isinstance(hp_space, ArchitectureHyperparameterSpace):
            assert self.is_architecture_trial
            return hp_space.bayesian_params(
                {
                    "conv1_filters": self.conv1_filters,
                    "conv1_kernel": self.conv1_kernel,
                    "conv2_filters": self.conv2_filters,
                    "conv2_kernel": self.conv2_kernel,
                    "conv3_filters": self.conv3_filters,
                    "conv3_kernel": self.conv3_kernel,
                    "conv4_filters": self.conv4_filters,
                    "conv4_kernel": self.conv4_kernel,
                    "pool_size": self.pool_size,
                    "fc1_size": self.fc1_size,
                    "fc2_size": self.fc2_size,
                }
            )

        batch_index = min(
            range(len(hp_space.batch_sizes)),
            key=lambda i: abs(hp_space.batch_sizes[i] - self.batch_size),
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
            "cls_pos_weight": self.cls_pos_weight,
            "batch_idx": float(batch_index),
        }

    def _training_env_keys(self) -> dict[str, Any]:
        return {
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
            "CLS_POS_WEIGHT": str(self.cls_pos_weight),
        }

    def _architecture_env_keys(self) -> dict[str, Any]:
        assert self.is_architecture_trial
        return {
            "CONV1_FILTERS": str(self.conv1_filters),
            "CONV1_KERNEL": str(self.conv1_kernel),
            "CONV1_PADDING": str(self.conv1_padding),
            "CONV2_FILTERS": str(self.conv2_filters),
            "CONV2_KERNEL": str(self.conv2_kernel),
            "CONV2_PADDING": str(self.conv2_padding),
            "CONV3_FILTERS": str(self.conv3_filters),
            "CONV3_KERNEL": str(self.conv3_kernel),
            "CONV3_PADDING": str(self.conv3_padding),
            "CONV4_FILTERS": str(self.conv4_filters),
            "CONV4_KERNEL": str(self.conv4_kernel),
            "CONV4_PADDING": str(self.conv4_padding),
            "POOL_SIZE": str(self.pool_size),
            "FC1_SIZE": str(self.fc1_size),
            "FC2_SIZE": str(self.fc2_size),
        }

    def trial_env_keys(self) -> dict[str, Any]:
        keys: dict[str, Any] = {
            "JOB_ID": self.trial_id,
            "PROG_DIR": str(self.dir_path),
        }
        if self.is_architecture_trial:
            keys.update(fixed_training_env_keys())
            keys.update(self._architecture_env_keys())
        else:
            keys.update(self._training_env_keys())
        return keys

    def create_files(self, *, env_lines: list[str] | None = None) -> str:
        self.dir_path.mkdir(parents=True, exist_ok=True)
        write_env(str(self.dir_path / ".env"), self.trial_env_keys(), env_lines)
        return self.trial_id
