"""Hyperparameter trial model (Pydantic), ORM bridge, and ``run.sh`` fragments."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from model.trial_status import TrialStatus
from util.hptune import write_env

TrialSignature = tuple[str, ...]


class HPTuneTrial(BaseModel):
    """One hyperparameter trial: training hparams, log status, identity.

    ``trial_id`` is the primary key in ``trials.csv``.
    """

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
    smoothing_divisor: int = 200
    conv1_filters: int = 16
    conv1_kernel: int = 9
    conv1_padding: int = 4
    conv2_filters: int = 32
    conv2_kernel: int = 5
    conv2_padding: int = 2
    conv3_filters: int = 64
    conv3_kernel: int = 3
    conv3_padding: int = 1
    conv4_filters: int = 128
    conv4_kernel: int = 3
    conv4_padding: int = 1
    pool_size: int = 4
    fc1_size: int = 120
    fc2_size: int = 60
    score: float = -1.0
    recall: float = -1.0
    precision: float = -1.0
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
        return self.proposed_signature(self.model_dump())

    @staticmethod
    def proposed_signature(d: Mapping[str, Any]) -> TrialSignature:
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
            f"{float(d['cls_pos_weight']):.12g}",
            int(d.get("smoothing_divisor", 200)),
            int(d.get("conv1_filters", 16)),
            int(d.get("conv1_kernel", 9)),
            int(d.get("conv1_padding", 4)),
            int(d.get("conv2_filters", 32)),
            int(d.get("conv2_kernel", 5)),
            int(d.get("conv2_padding", 2)),
            int(d.get("conv3_filters", 64)),
            int(d.get("conv3_kernel", 3)),
            int(d.get("conv3_padding", 1)),
            int(d.get("conv4_filters", 128)),
            int(d.get("conv4_kernel", 3)),
            int(d.get("conv4_padding", 1)),
            int(d.get("pool_size", 4)),
            int(d.get("fc1_size", 120)),
            int(d.get("fc2_size", 60)),
        )

    def log_pass_hyperparameters(self, *, context: str) -> None:
        logger.opt(lazy=True).info(
            "Hyperparameters for this pass ({ctx}): trial_id={id} lr={lr:.2e} epochs={ep} "
            "dropout={do:.4f} weight_decay={wd:.2e} batch_size={bs} gradient_clip={gc:.3f} "
            "lr_scheduler={ls} lr_scheduler_factor={lf:.3f} lr_scheduler_patience={lp} "
            "early_stopping_patience={esp} cls_pos_weight={cpw:.3f} "
            "smoothing_divisor={sd} "
            "conv1=({c1f},{c1k},{c1p}) conv2=({c2f},{c2k},{c2p}) "
            "conv3=({c3f},{c3k},{c3p}) conv4=({c4f},{c4k},{c4p}) "
            "pool_size={pool} fc1={fc1} fc2={fc2}",
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
            sd=lambda: self.smoothing_divisor,
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

    def _conv_filter_index(
        self, filters: int, allowed: tuple[int, ...]
    ) -> float:
        return float(
            min(range(len(allowed)), key=lambda i: abs(allowed[i] - filters))
        )

    def _kernel_index(self, kernel: int, allowed: tuple[int, ...]) -> float:
        return float(
            min(range(len(allowed)), key=lambda i: abs(allowed[i] - kernel))
        )

    def bayesian_params(
        self,
        batch_sizes: tuple[int, ...],
        *,
        allowed_conv_filters: tuple[int, ...],
        allowed_kernels: tuple[int, ...],
        allowed_pool_sizes: tuple[int, ...],
    ) -> dict[str, float]:
        """Float-only parameter dict aligned with BayesianOptimization ``pbounds``."""
        batch_index = min(
            range(len(batch_sizes)),
            key=lambda i: abs(batch_sizes[i] - self.batch_size),
        )
        pool_index = min(
            range(len(allowed_pool_sizes)),
            key=lambda i: abs(allowed_pool_sizes[i] - self.pool_size),
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
            "smoothing_divisor": float(self.smoothing_divisor),
            "conv1_f_idx": self._conv_filter_index(
                self.conv1_filters, allowed_conv_filters
            ),
            "conv2_f_idx": self._conv_filter_index(
                self.conv2_filters, allowed_conv_filters
            ),
            "conv3_f_idx": self._conv_filter_index(
                self.conv3_filters, allowed_conv_filters
            ),
            "conv4_f_idx": self._conv_filter_index(
                self.conv4_filters, allowed_conv_filters
            ),
            "conv1_k_idx": self._kernel_index(self.conv1_kernel, allowed_kernels),
            "conv2_k_idx": self._kernel_index(self.conv2_kernel, allowed_kernels),
            "conv3_k_idx": self._kernel_index(self.conv3_kernel, allowed_kernels),
            "conv4_k_idx": self._kernel_index(self.conv4_kernel, allowed_kernels),
            "pool_idx": float(pool_index),
            "fc1": float(self.fc1_size),
            "fc2": float(self.fc2_size),
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
            "CLS_POS_WEIGHT": str(self.cls_pos_weight),
            "SMOOTHING_DIVISOR": str(self.smoothing_divisor),
            "DATA_PATH": str(self.dir_path / "processed_dataset.pt"),
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
            "PROG_DIR": self.dir_path,
        }

    def create_files(self, *, env_lines: list[str] | None = None) -> str:
        self.dir_path.mkdir(parents=True, exist_ok=True)
        write_env(str(self.dir_path / ".env"), self.trial_env_keys(), env_lines)
        return self.trial_id
