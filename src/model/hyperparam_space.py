"""Bayesian hyperparameter tuning orchestration (trial log, acquisition, trial dirs)."""

from __future__ import annotations
from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict


class HyperparameterSpace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    allowed_epochs: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    bounds: dict[str, tuple[float, float]]

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
