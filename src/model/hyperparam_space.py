"""Bayesian hyperparameter tuning orchestration (trial log, acquisition, trial dirs)."""

from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from util.data_loading import env_float, env_int, env_tuple


def _pick_index(values: tuple[int, ...], idx: float) -> int:
    return values[int(np.clip(round(idx), 0, len(values) - 1))]


def _kernel_padding(kernel: int) -> int:
    return max(0, kernel // 2)


class HyperparameterSpace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    allowed_epochs: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    allowed_conv_filters: tuple[int, ...]
    allowed_kernels: tuple[int, ...]
    allowed_pool_sizes: tuple[int, ...]
    num_initial_trials: int = Field(ge=1)
    random_insert_every: int = Field(ge=0)
    expected_improvement_xi: float = Field(ge=0)
    bounds: dict[str, tuple[float, float]]

    @staticmethod
    def from_env() -> HyperparameterSpace:
        allowed_epochs = env_tuple("HPTUNE_ALLOWED_EPOCHS")
        batch_sizes = env_tuple("HPTUNE_ALLOWED_BATCH_SIZES")
        allowed_conv_filters = env_tuple("HPTUNE_ALLOWED_CONV_FILTERS")
        allowed_kernels = env_tuple("HPTUNE_ALLOWED_KERNELS")
        allowed_pool_sizes = env_tuple("HPTUNE_ALLOWED_POOL_SIZES")
        num_initial_trials = env_int("HPTUNE_NUM_INITIAL_TRIALS")
        random_insert_every = env_int("HPTUNE_RANDOM_INSERT_EVERY")
        expected_improvement_xi = float(os.environ["HPTUNE_EI_XI"])

        n_filters = len(allowed_conv_filters) - 1
        n_kernels = len(allowed_kernels) - 1
        n_pool = len(allowed_pool_sizes) - 1

        bounds = {
            "lr": (env_float("HPTUNE_LR_MIN"), env_float("HPTUNE_LR_MAX")),
            "dropout": (
                env_float("HPTUNE_DROPOUT_MIN"),
                env_float("HPTUNE_DROPOUT_MAX"),
            ),
            "log_wd": (
                env_float("HPTUNE_WEIGHT_DECAY_LOG_MIN"),
                env_float("HPTUNE_WEIGHT_DECAY_LOG_MAX"),
            ),
            "gradient_clip": (
                env_float("HPTUNE_GRADIENT_CLIP_MIN"),
                env_float("HPTUNE_GRADIENT_CLIP_MAX"),
            ),
            "lr_scheduler_factor": (
                env_float("HPTUNE_LR_SCHEDULER_FACTOR_MIN"),
                env_float("HPTUNE_LR_SCHEDULER_FACTOR_MAX"),
            ),
            "lr_sched_patience": (
                env_float("HPTUNE_LR_SCHEDULER_PATIENCE_MIN"),
                env_float("HPTUNE_LR_SCHEDULER_PATIENCE_MAX"),
            ),
            "early_stop_patience": (
                env_float("HPTUNE_EARLY_STOPPING_PATIENCE_MIN"),
                env_float("HPTUNE_EARLY_STOPPING_PATIENCE_MAX"),
            ),
            "cls_pos_weight": (
                env_float("HPTUNE_CLS_POS_WEIGHT_MIN"),
                env_float("HPTUNE_CLS_POS_WEIGHT_MAX"),
            ),
            "smoothing_divisor": (
                env_float("HPTUNE_SMOOTHING_DIVISOR_MIN"),
                env_float("HPTUNE_SMOOTHING_DIVISOR_MAX"),
            ),
            "conv1_f_idx": (0.0, float(n_filters)),
            "conv2_f_idx": (0.0, float(n_filters)),
            "conv3_f_idx": (0.0, float(n_filters)),
            "conv4_f_idx": (0.0, float(n_filters)),
            "conv1_k_idx": (0.0, float(n_kernels)),
            "conv2_k_idx": (0.0, float(n_kernels)),
            "conv3_k_idx": (0.0, float(n_kernels)),
            "conv4_k_idx": (0.0, float(n_kernels)),
            "pool_idx": (0.0, float(n_pool)),
            "fc1": (env_float("HPTUNE_FC1_MIN"), env_float("HPTUNE_FC1_MAX")),
            "fc2": (env_float("HPTUNE_FC2_MIN"), env_float("HPTUNE_FC2_MAX")),
            "epochs": (float(min(allowed_epochs)), float(max(allowed_epochs))),
            "batch_idx": (0.0, float(len(batch_sizes) - 1)),
            "lr_scheduler_u": (0.0, 1.0),
        }

        return HyperparameterSpace(
            allowed_epochs=allowed_epochs,
            batch_sizes=batch_sizes,
            allowed_conv_filters=allowed_conv_filters,
            allowed_kernels=allowed_kernels,
            allowed_pool_sizes=allowed_pool_sizes,
            num_initial_trials=num_initial_trials,
            random_insert_every=random_insert_every,
            expected_improvement_xi=expected_improvement_xi,
            bounds=bounds,
        )

    def _architecture_from_indices(self, s: Mapping[str, float]) -> dict[str, int]:
        conv1_kernel = _pick_index(self.allowed_kernels, s["conv1_k_idx"])
        conv2_kernel = _pick_index(self.allowed_kernels, s["conv2_k_idx"])
        conv3_kernel = _pick_index(self.allowed_kernels, s["conv3_k_idx"])
        conv4_kernel = _pick_index(self.allowed_kernels, s["conv4_k_idx"])
        return {
            "conv1_filters": _pick_index(self.allowed_conv_filters, s["conv1_f_idx"]),
            "conv1_kernel": conv1_kernel,
            "conv1_padding": _kernel_padding(conv1_kernel),
            "conv2_filters": _pick_index(self.allowed_conv_filters, s["conv2_f_idx"]),
            "conv2_kernel": conv2_kernel,
            "conv2_padding": _kernel_padding(conv2_kernel),
            "conv3_filters": _pick_index(self.allowed_conv_filters, s["conv3_f_idx"]),
            "conv3_kernel": conv3_kernel,
            "conv3_padding": _kernel_padding(conv3_kernel),
            "conv4_filters": _pick_index(self.allowed_conv_filters, s["conv4_f_idx"]),
            "conv4_kernel": conv4_kernel,
            "conv4_padding": _kernel_padding(conv4_kernel),
            "pool_size": _pick_index(self.allowed_pool_sizes, s["pool_idx"]),
            "fc1_size": int(round(s["fc1"])),
            "fc2_size": int(round(s["fc2"])),
        }

    # -----------------------------
    # Sampling
    # -----------------------------

    def sample_random(self) -> dict[str, Any]:
        b = self.bounds
        log_uniform = lambda lo, hi: 10 ** np.random.uniform(np.log10(lo), np.log10(hi))
        bo_sample = {
            "lr": log_uniform(*b["lr"]),
            "epochs": float(np.random.choice(self.allowed_epochs)),
            "dropout": float(np.random.uniform(*b["dropout"])),
            "log_wd": float(np.random.uniform(*b["log_wd"])),
            "batch_idx": float(np.random.randint(0, len(self.batch_sizes))),
            "gradient_clip": float(np.random.uniform(*b["gradient_clip"])),
            "lr_scheduler_u": float(np.random.rand()),
            "lr_scheduler_factor": float(np.random.uniform(*b["lr_scheduler_factor"])),
            "lr_sched_patience": float(
                np.random.randint(*map(int, b["lr_sched_patience"]))
            ),
            "early_stop_patience": float(
                np.random.randint(*map(int, b["early_stop_patience"]))
            ),
            "cls_pos_weight": float(np.random.uniform(*b["cls_pos_weight"])),
            "smoothing_divisor": float(
                np.random.randint(
                    int(b["smoothing_divisor"][0]),
                    int(b["smoothing_divisor"][1]) + 1,
                )
            ),
            "conv1_f_idx": float(np.random.randint(0, len(self.allowed_conv_filters))),
            "conv2_f_idx": float(np.random.randint(0, len(self.allowed_conv_filters))),
            "conv3_f_idx": float(np.random.randint(0, len(self.allowed_conv_filters))),
            "conv4_f_idx": float(np.random.randint(0, len(self.allowed_conv_filters))),
            "conv1_k_idx": float(np.random.randint(0, len(self.allowed_kernels))),
            "conv2_k_idx": float(np.random.randint(0, len(self.allowed_kernels))),
            "conv3_k_idx": float(np.random.randint(0, len(self.allowed_kernels))),
            "conv4_k_idx": float(np.random.randint(0, len(self.allowed_kernels))),
            "pool_idx": float(np.random.randint(0, len(self.allowed_pool_sizes))),
            "fc1": float(np.random.uniform(*b["fc1"])),
            "fc2": float(np.random.uniform(*b["fc2"])),
        }
        return self.suggestion_to_trial(bo_sample)

    def suggestion_to_trial(self, s: Mapping[str, float]) -> dict[str, Any]:
        bi = int(np.clip(round(s["batch_idx"]), 0, len(self.batch_sizes) - 1))
        arch = self._architecture_from_indices(s)

        return {
            "lr": float(s["lr"]),
            "epochs": min(self.allowed_epochs, key=lambda x: abs(x - s["epochs"])),
            "dropout": float(s["dropout"]),
            "weight_decay": 10 ** float(s["log_wd"]),
            "batch_size": self.batch_sizes[bi],
            "gradient_clip": float(s["gradient_clip"]),
            "lr_scheduler": s["lr_scheduler_u"] >= 0.5,
            "lr_scheduler_factor": float(s["lr_scheduler_factor"]),
            "lr_scheduler_patience": int(round(s["lr_sched_patience"])),
            "early_stopping_patience": int(round(s["early_stop_patience"])),
            "cls_pos_weight": float(s["cls_pos_weight"]),
            "smoothing_divisor": int(
                round(
                    np.clip(
                        s["smoothing_divisor"],
                        *self.bounds["smoothing_divisor"],
                    )
                )
            ),
            **arch,
        }
