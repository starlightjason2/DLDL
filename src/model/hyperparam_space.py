"""Bayesian hyperparameter tuning orchestration (trial log, acquisition, trial dirs)."""

from __future__ import annotations

import os
from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from util.data_loading import env_tuple, env_float, env_int


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
        allowed_epochs = env_tuple("HPTUNE_ALLOWED_EPOCHS")
        batch_sizes = env_tuple("HPTUNE_ALLOWED_BATCH_SIZES")
        num_initial_trials = env_int("HPTUNE_NUM_INITIAL_TRIALS")
        random_insert_every = env_int("HPTUNE_RANDOM_INSERT_EVERY")
        expected_improvement_xi = float(os.environ["HPTUNE_EI_XI"])
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

    # -----------------------------
    # Sampling
    # -----------------------------

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
        }

    def suggestion_to_trial(self, s: dict[str, float]) -> dict[str, Any]:
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
        }
