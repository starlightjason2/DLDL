"""``Settings.load()`` → JSON config + merged training/architecture (file + env overrides)."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from config.schema import ArchitectureConfig, DldlConfigFile, TrainingConfig


@functools.lru_cache(maxsize=1)
def _dotenv() -> None:
    p = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(p)


class Settings(BaseModel):
    """File-backed config + env-overridden training/architecture; paths use ``os.environ`` directly elsewhere."""

    model_config = ConfigDict(frozen=True)

    project_root: str
    cfg: DldlConfigFile
    dldl_config_path: str
    training_config: TrainingConfig
    architecture_config: ArchitectureConfig

    @classmethod
    @functools.lru_cache(maxsize=1)
    def load(cls) -> "Settings":
        _dotenv()

        root = Path(__file__).resolve().parents[2]
        raw = os.environ.get("DLDL_CONFIG")
        cfg_path = (
            Path(raw).resolve()
            if raw and os.path.isabs(raw)
            else (
                (root / raw).resolve()
                if raw
                else Path(__file__).resolve().parent / "dldl.json"
            )
        )
        if not cfg_path.is_file():
            raise FileNotFoundError(f"DLDL config missing: {cfg_path}")

        cfg = DldlConfigFile.model_validate(
            json.loads(cfg_path.read_text(encoding="utf-8"))
        )
        r = str(root)
        tr, ar = TrainingConfig.merge_env(
            cfg.default_training
        ), ArchitectureConfig.merge_env(cfg.architecture)
        return cls(
            project_root=r,
            cfg=cfg,
            dldl_config_path=str(cfg_path),
            training_config=tr,
            architecture_config=ar,
        )

    def default_hptune_param_bounds(
        self,
        allowed_epochs: Optional[Tuple[int, ...]] = None,
        batch_sizes: Optional[Tuple[int, ...]] = None,
    ) -> dict[str, Tuple[float, float]]:
        h, eps, bs = self.cfg.hptune, allowed_epochs, batch_sizes
        eps = tuple(eps) if eps is not None else tuple(h.allowed_epochs)
        bs = tuple(bs) if bs is not None else tuple(h.allowed_batch_sizes)
        n = max(len(bs) - 1, 0)
        return {
            "lr": (h.lr_min, h.lr_max),
            "dropout": (h.dropout_min, h.dropout_max),
            "log_wd": (h.weight_decay_log_min, h.weight_decay_log_max),
            "epochs": (float(min(eps)), float(max(eps))),
            "gradient_clip": (h.gradient_clip_min, h.gradient_clip_max),
            "lr_scheduler_u": (0.0, 1.0),
            "lr_scheduler_factor": (
                h.lr_scheduler_factor_min,
                h.lr_scheduler_factor_max,
            ),
            "lr_sched_patience": (
                float(h.lr_scheduler_patience_min),
                float(h.lr_scheduler_patience_max),
            ),
            "early_stop_patience": (
                float(h.early_stopping_patience_min),
                float(h.early_stopping_patience_max),
            ),
            "batch_idx": (0.0, float(n)),
        }
