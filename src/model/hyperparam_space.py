"""Bayesian hyperparameter search spaces for training and architecture tuning."""

from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from util.data_loading import env_float, env_int, env_tuple


def _same_padding(kernel: int) -> int:
    return kernel // 2


class HyperparameterSpace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    allowed_epochs: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    num_initial_trials: int = Field(ge=1)
    random_insert_every: int = Field(ge=0)
    expected_improvement_xi: float = Field(ge=0)
    bounds: dict[str, tuple[float, float]]

    @staticmethod
    def from_env() -> HyperparameterSpace:
        allowed_epochs = env_tuple("HP_TUNE_ALLOWED_EPOCHS")
        batch_sizes = env_tuple("HP_TUNE_ALLOWED_BATCH_SIZES")
        num_initial_trials = env_int("HP_TUNE_NUM_INITIAL_TRIALS")
        random_insert_every = env_int("HP_TUNE_RANDOM_INSERT_EVERY")
        expected_improvement_xi = float(os.environ["HP_TUNE_EI_XI"])
        bounds = {
            "lr": (env_float("HP_TUNE_LR_MIN"), env_float("HP_TUNE_LR_MAX")),
            "dropout": (
                env_float("HP_TUNE_DROPOUT_MIN"),
                env_float("HP_TUNE_DROPOUT_MAX"),
            ),
            "log_wd": (
                env_float("HP_TUNE_WEIGHT_DECAY_LOG_MIN"),
                env_float("HP_TUNE_WEIGHT_DECAY_LOG_MAX"),
            ),
            "gradient_clip": (
                env_float("HP_TUNE_GRADIENT_CLIP_MIN"),
                env_float("HP_TUNE_GRADIENT_CLIP_MAX"),
            ),
            "lr_scheduler_factor": (
                env_float("HP_TUNE_LR_SCHEDULER_FACTOR_MIN"),
                env_float("HP_TUNE_LR_SCHEDULER_FACTOR_MAX"),
            ),
            "lr_sched_patience": (
                env_float("HP_TUNE_LR_SCHEDULER_PATIENCE_MIN"),
                env_float("HP_TUNE_LR_SCHEDULER_PATIENCE_MAX"),
            ),
            "early_stop_patience": (
                env_float("HP_TUNE_EARLY_STOPPING_PATIENCE_MIN"),
                env_float("HP_TUNE_EARLY_STOPPING_PATIENCE_MAX"),
            ),
            "cls_pos_weight": (
                env_float("HP_TUNE_CLS_POS_WEIGHT_MIN"),
                env_float("HP_TUNE_CLS_POS_WEIGHT_MAX"),
            ),
            "epochs": (float(min(allowed_epochs)), float(max(allowed_epochs))),
            "batch_idx": (0.0, float(len(batch_sizes) - 1)),
            "lr_scheduler_u": (0.0, 1.0),
        }

        return HyperparameterSpace(
            allowed_epochs=allowed_epochs,
            batch_sizes=batch_sizes,
            num_initial_trials=num_initial_trials,
            random_insert_every=random_insert_every,
            expected_improvement_xi=expected_improvement_xi,
            bounds=bounds,
        )

    def sample_random(self) -> dict[str, Any]:
        b = self.bounds
        log_uniform = lambda lo, hi: 10 ** np.random.uniform(np.log10(lo), np.log10(hi))

        return {
            "lr": log_uniform(*b["lr"]),
            "epochs": int(np.random.choice(self.allowed_epochs)),
            "dropout": float(np.random.uniform(*b["dropout"])),
            "weight_decay": 10 ** float(np.random.uniform(*b["log_wd"])),
            "batch_size": int(np.random.choice(self.batch_sizes)),
            "gradient_clip": float(np.random.uniform(*b["gradient_clip"])),
            "lr_scheduler": np.random.rand() > 0.5,
            "lr_scheduler_factor": float(np.random.uniform(*b["lr_scheduler_factor"])),
            "lr_scheduler_patience": int(
                np.random.randint(*map(int, b["lr_sched_patience"]))
            ),
            "early_stopping_patience": int(
                np.random.randint(*map(int, b["early_stop_patience"]))
            ),
            "cls_pos_weight": float(np.random.uniform(*b["cls_pos_weight"])),
        }

    def suggestion_to_trial(self, s: Mapping[str, float]) -> dict[str, Any]:
        bi = int(np.clip(round(s["batch_idx"]), 0, len(self.batch_sizes) - 1))

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
        }


class ArchitectureHyperparameterSpace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conv_filters: tuple[int, ...]
    kernels: tuple[int, ...]
    pool_sizes: tuple[int, ...]
    num_initial_trials: int = Field(ge=1)
    random_insert_every: int = Field(ge=0)
    expected_improvement_xi: float = Field(ge=0)
    fc1_min: int = Field(ge=1)
    fc1_max: int = Field(ge=1)
    fc2_min: int = Field(ge=1)
    fc2_max: int = Field(ge=1)
    bounds: dict[str, tuple[float, float]]

    @staticmethod
    def from_env() -> ArchitectureHyperparameterSpace:
        conv_filters = env_tuple("ARCH_TUNE_CONV_FILTERS")
        kernels = env_tuple("ARCH_TUNE_KERNELS")
        pool_sizes = env_tuple("ARCH_TUNE_POOL_SIZES")
        fc1_min = env_int("ARCH_TUNE_FC1_MIN")
        fc1_max = env_int("ARCH_TUNE_FC1_MAX")
        fc2_min = env_int("ARCH_TUNE_FC2_MIN")
        fc2_max = env_int("ARCH_TUNE_FC2_MAX")
        if fc1_min > fc1_max:
            raise ValueError("ARCH_TUNE_FC1_MIN must be <= ARCH_TUNE_FC1_MAX")
        if fc2_min > fc2_max:
            raise ValueError("ARCH_TUNE_FC2_MIN must be <= ARCH_TUNE_FC2_MAX")

        def _idx_bounds(count: int) -> tuple[float, float]:
            return (0.0, float(max(count - 1, 0)))

        bounds = {
            "conv1_f_idx": _idx_bounds(len(conv_filters)),
            "conv2_f_idx": _idx_bounds(len(conv_filters)),
            "conv3_f_idx": _idx_bounds(len(conv_filters)),
            "conv4_f_idx": _idx_bounds(len(conv_filters)),
            "conv1_k_idx": _idx_bounds(len(kernels)),
            "conv2_k_idx": _idx_bounds(len(kernels)),
            "conv3_k_idx": _idx_bounds(len(kernels)),
            "conv4_k_idx": _idx_bounds(len(kernels)),
            "pool_idx": _idx_bounds(len(pool_sizes)),
            "fc1": (float(fc1_min), float(fc1_max)),
            "fc2": (float(fc2_min), float(fc2_max)),
        }

        return ArchitectureHyperparameterSpace(
            conv_filters=conv_filters,
            kernels=kernels,
            pool_sizes=pool_sizes,
            num_initial_trials=env_int("ARCH_TUNE_NUM_INITIAL_TRIALS"),
            random_insert_every=env_int("ARCH_TUNE_RANDOM_INSERT_EVERY"),
            expected_improvement_xi=env_float("ARCH_TUNE_EI_XI"),
            fc1_min=fc1_min,
            fc1_max=fc1_max,
            fc2_min=fc2_min,
            fc2_max=fc2_max,
            bounds=bounds,
        )

    @staticmethod
    def _pick_index(options: tuple[int, ...], raw: float) -> int:
        return int(np.clip(round(raw), 0, len(options) - 1))

    def _resolve_architecture(self, s: dict[str, float]) -> dict[str, int]:
        conv1_kernel = self.kernels[self._pick_index(self.kernels, s["conv1_k_idx"])]
        conv2_kernel = self.kernels[self._pick_index(self.kernels, s["conv2_k_idx"])]
        conv3_kernel = self.kernels[self._pick_index(self.kernels, s["conv3_k_idx"])]
        conv4_kernel = self.kernels[self._pick_index(self.kernels, s["conv4_k_idx"])]

        fc1_size = int(np.clip(round(s["fc1"]), self.fc1_min, self.fc1_max))
        fc2_size = int(
            np.clip(round(s["fc2"]), self.fc2_min, min(self.fc2_max, fc1_size))
        )

        return {
            "conv1_filters": self.conv_filters[
                self._pick_index(self.conv_filters, s["conv1_f_idx"])
            ],
            "conv1_kernel": conv1_kernel,
            "conv1_padding": _same_padding(conv1_kernel),
            "conv2_filters": self.conv_filters[
                self._pick_index(self.conv_filters, s["conv2_f_idx"])
            ],
            "conv2_kernel": conv2_kernel,
            "conv2_padding": _same_padding(conv2_kernel),
            "conv3_filters": self.conv_filters[
                self._pick_index(self.conv_filters, s["conv3_f_idx"])
            ],
            "conv3_kernel": conv3_kernel,
            "conv3_padding": _same_padding(conv3_kernel),
            "conv4_filters": self.conv_filters[
                self._pick_index(self.conv_filters, s["conv4_f_idx"])
            ],
            "conv4_kernel": conv4_kernel,
            "conv4_padding": _same_padding(conv4_kernel),
            "pool_size": self.pool_sizes[
                self._pick_index(self.pool_sizes, s["pool_idx"])
            ],
            "fc1_size": fc1_size,
            "fc2_size": fc2_size,
        }

    def sample_random(self) -> dict[str, Any]:
        sample = {
            key: float(np.random.uniform(lo, hi))
            for key, (lo, hi) in self.bounds.items()
        }
        return self.suggestion_to_trial(sample)

    def suggestion_to_trial(self, s: Mapping[str, float]) -> dict[str, Any]:
        return self._resolve_architecture(dict(s))

    def bayesian_params(self, arch: Mapping[str, int]) -> dict[str, float]:
        def _index(options: tuple[int, ...], value: int) -> float:
            return float(
                min(range(len(options)), key=lambda i: abs(options[i] - value))
            )

        return {
            "conv1_f_idx": _index(self.conv_filters, int(arch["conv1_filters"])),
            "conv2_f_idx": _index(self.conv_filters, int(arch["conv2_filters"])),
            "conv3_f_idx": _index(self.conv_filters, int(arch["conv3_filters"])),
            "conv4_f_idx": _index(self.conv_filters, int(arch["conv4_filters"])),
            "conv1_k_idx": _index(self.kernels, int(arch["conv1_kernel"])),
            "conv2_k_idx": _index(self.kernels, int(arch["conv2_kernel"])),
            "conv3_k_idx": _index(self.kernels, int(arch["conv3_kernel"])),
            "conv4_k_idx": _index(self.kernels, int(arch["conv4_kernel"])),
            "pool_idx": _index(self.pool_sizes, int(arch["pool_size"])),
            "fc1": float(arch["fc1_size"]),
            "fc2": float(arch["fc2_size"]),
        }


def hp_tune_mode() -> str:
    mode = os.environ.get("HP_TUNE_MODE", "training").strip().lower()
    if mode not in {"training", "architecture"}:
        raise ValueError(f"HP_TUNE_MODE must be 'training' or 'architecture', got {mode!r}")
    return mode
