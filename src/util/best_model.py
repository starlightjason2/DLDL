"""Load the canonical model from ``best_model/`` at the repo root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from model.cnn import IpCNN
    from model.dataset import IpDataset

BEST_MODEL_DIRNAME = "best_model"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def best_model_dir() -> Path:
    return repo_root() / BEST_MODEL_DIRNAME


def load_best_model_env() -> Path:
    """Load project ``.env``, then ``best_model/.env`` overrides."""
    root = repo_root()
    load_dotenv(root / ".env", encoding="utf-8")
    model_dir = best_model_dir()
    load_dotenv(model_dir / ".env", encoding="utf-8", override=True)
    return model_dir


def resolve_best_model_checkpoint() -> Path | None:
    checkpoints = sorted(best_model_dir().glob("*_best_params.pt"))
    return checkpoints[0] if checkpoints else None


def load_best_model_cnn(dataset: "IpDataset") -> "IpCNN | None":
    """Build an eval-mode ``IpCNN`` from ``best_model/*_best_params.pt``."""
    from util.training import build_cnn_from_env, load_checkpoint_into_model

    load_best_model_env()
    checkpoint = resolve_best_model_checkpoint()
    if checkpoint is None:
        return None

    model_dir = best_model_dir()
    os.environ["PROG_DIR"] = str(model_dir)
    model = build_cnn_from_env(dataset, str(model_dir))
    load_checkpoint_into_model(model, checkpoint)
    model.eval()
    return model
