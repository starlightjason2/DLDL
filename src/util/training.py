"""Training run artifacts: best-epoch snapshot and checkpoint loading."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from model.cnn import IpCNN
    from model.dataset import IpDataset

_TRAINING_ENV_KEYS = (
    "JOB_ID",
    "LEARNING_RATE",
    "NUM_EPOCHS",
    "BATCH_SIZE",
    "DROPOUT_RATE",
    "WEIGHT_DECAY",
    "GRADIENT_CLIP",
    "LR_SCHEDULER",
    "LR_SCHEDULER_FACTOR",
    "LR_SCHEDULER_PATIENCE",
    "EARLY_STOPPING_PATIENCE",
    "CLS_POS_WEIGHT",
    "DECISION_THRESHOLD",
    "MIN_PRECISION",
    "FBETA",
    "NORMALIZATION_TYPE",
    "DATA_PATH",
    "TRAIN_LABELS_PATH",
    "CONV1_FILTERS",
    "CONV1_KERNEL",
    "CONV1_PADDING",
    "CONV2_FILTERS",
    "CONV2_KERNEL",
    "CONV2_PADDING",
    "CONV3_FILTERS",
    "CONV3_KERNEL",
    "CONV3_PADDING",
    "CONV4_FILTERS",
    "CONV4_KERNEL",
    "CONV4_PADDING",
    "POOL_SIZE",
    "FC1_SIZE",
    "FC2_SIZE",
    "PROG_DIR",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_prog_dir(prog_dir: str | Path | None = None) -> Path:
    root = prog_dir if prog_dir is not None else os.environ["PROG_DIR"]
    path = Path(root)
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def training_env_keys(job_id: str, prog_dir: Path) -> dict[str, str]:
    """Env vars persisted in ``best_epoch/.env`` for reproducible reload."""
    keys = {
        key: os.environ[key]
        for key in _TRAINING_ENV_KEYS
        if key in os.environ and key not in {"JOB_ID", "PROG_DIR"}
    }
    keys["JOB_ID"] = job_id
    keys["PROG_DIR"] = str(prog_dir)
    return keys


def build_cnn_from_env(dataset: "IpDataset", prog_dir: str) -> "IpCNN":
    from model.cnn import IpCNN

    def _conv(prefix: str) -> tuple[int, int, int]:
        return (
            int(os.environ[f"{prefix}_FILTERS"]),
            int(os.environ[f"{prefix}_KERNEL"]),
            int(os.environ[f"{prefix}_PADDING"]),
        )

    return IpCNN(
        dataset,
        prog_dir=prog_dir,
        conv1=_conv("CONV1"),
        conv2=_conv("CONV2"),
        conv3=_conv("CONV3"),
        conv4=_conv("CONV4"),
        pool_size=int(os.environ["POOL_SIZE"]),
        fc1_size=int(os.environ["FC1_SIZE"]),
        fc2_size=int(os.environ["FC2_SIZE"]),
        dropout_rate=float(os.environ["DROPOUT_RATE"]),
        cls_pos_weight=float(os.environ["CLS_POS_WEIGHT"]),
        decision_threshold=float(os.environ["DECISION_THRESHOLD"]),
    )


def _companion_threshold_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(
        checkpoint_path.name.replace("_best_params.pt", "_best_threshold.txt")
    )


def load_checkpoint_into_model(model: "IpCNN", checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    threshold_path = _companion_threshold_path(checkpoint_path)
    if threshold_path.exists():
        model.decision_threshold = float(
            threshold_path.read_text(encoding="utf-8").strip()
        )


def sync_best_epoch_artifacts(job_id: str, prog_dir: str | Path) -> None:
    """Refresh ``<prog_dir>/best_epoch/`` with the run's best checkpoint and config.

    Mirrors ``sync_best_trial_artifacts``: keeps a single canonical checkpoint,
    training log, and ``.env`` snapshot under ``best_epoch/``.
    """
    root = _resolve_prog_dir(prog_dir)
    dest = root / "best_epoch"
    dest.mkdir(parents=True, exist_ok=True)

    checkpoint_src = root / f"{job_id}_best_params.pt"
    if not checkpoint_src.exists():
        logger.warning(
            "Best-epoch sync: no checkpoint at {} — skipping snapshot",
            checkpoint_src,
        )
        return

    for existing in dest.glob("*_best_params.pt"):
        existing.unlink()
    shutil.copy2(checkpoint_src, dest / checkpoint_src.name)

    threshold_src = _companion_threshold_path(checkpoint_src)
    if threshold_src.exists():
        for existing in dest.glob("*_best_threshold.txt"):
            existing.unlink()
        shutil.copy2(threshold_src, dest / threshold_src.name)

    log_src = root / f"{job_id}_training_log.csv"
    if log_src.exists():
        for existing in dest.glob("*_training_log.csv"):
            existing.unlink()
        shutil.copy2(log_src, dest / log_src.name)

    from util.hptune import write_env

    env_keys = training_env_keys(job_id, dest)
    if threshold_src.exists():
        env_keys["DECISION_THRESHOLD"] = threshold_src.read_text(
            encoding="utf-8"
        ).strip()

    write_env(
        str(dest / ".env"),
        env_keys,
        env_lines=[
            "# Best-epoch snapshot (see ``util.training.sync_best_epoch_artifacts``)"
        ],
    )

    logger.info(
        "Best-epoch snapshot updated: job_id={} -> {}",
        job_id,
        dest,
    )


def load_best_epoch_cnn(dataset: "IpDataset") -> "IpCNN | None":
    """Build an ``IpCNN`` from ``PROG_DIR/best_epoch/*_best_params.pt``."""
    best_dir = _resolve_prog_dir() / "best_epoch"
    checkpoints = sorted(best_dir.glob("*_best_params.pt"))
    if not checkpoints:
        return None

    model = build_cnn_from_env(dataset, str(best_dir))
    load_checkpoint_into_model(model, checkpoints[0])
    model.eval()
    return model
