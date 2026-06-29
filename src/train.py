"""Training entry point for DLDL disruption prediction model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from model.cnn import IpCNN
from model.dataset import IpDataset

_REPO = Path(__file__).resolve().parents[1]  # project root (src/..)
load_dotenv(dotenv_path=_REPO / ".env", encoding="utf-8")


def _abs(p: str) -> Path:
    """Resolve a path relative to the repo root if not already absolute."""
    return Path(p) if Path(p).is_absolute() else _REPO / p


def _configure_logging(prog_dir: Path, job_id: str) -> None:
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.remove()
    logger.add(sys.stderr, format=log_format, colorize=True, level="INFO")

    logger.add(
        prog_dir / f"{job_id}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
    )


def main() -> None:
    prog_dir = _abs(os.environ["PROG_DIR"])
    job_id = os.environ["JOB_ID"]
    data_path = _abs(os.environ["DATA_PATH"])
    labels_path = _abs(os.environ["TRAIN_LABELS_PATH"])

    # Ensure output directories exist before logging starts
    prog_dir.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    _configure_logging(prog_dir, job_id)

    lr_scheduler = os.environ["LR_SCHEDULER"].lower() in ("true", "1", "yes", "on")

    warm_start = os.environ.get("WARM_START_CHECKPOINT") or None

    dataset = IpDataset(
        normalization_type=os.environ["NORMALIZATION_TYPE"],
        data_file=str(data_path),
        labels_file=str(labels_path),
        labels_path=str(_abs(os.environ["LABELS_PATH"])),
        data_dir=str(_abs(os.environ["DATA_DIR"])),
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )

    model = IpCNN(
        dataset,
        prog_dir=str(prog_dir),
        conv1=(
            int(os.environ["CONV1_FILTERS"]),
            int(os.environ["CONV1_KERNEL"]),
            int(os.environ["CONV1_PADDING"]),
        ),
        conv2=(
            int(os.environ["CONV2_FILTERS"]),
            int(os.environ["CONV2_KERNEL"]),
            int(os.environ["CONV2_PADDING"]),
        ),
        conv3=(
            int(os.environ["CONV3_FILTERS"]),
            int(os.environ["CONV3_KERNEL"]),
            int(os.environ["CONV3_PADDING"]),
        ),
        conv4=(
            int(os.environ["CONV4_FILTERS"]),
            int(os.environ["CONV4_KERNEL"]),
            int(os.environ["CONV4_PADDING"]),
        ),
        pool_size=int(os.environ["POOL_SIZE"]),
        fc1_size=int(os.environ["FC1_SIZE"]),
        fc2_size=int(os.environ["FC2_SIZE"]),
        dropout_rate=float(os.environ["DROPOUT_RATE"]),
        cls_pos_weight=float(os.environ["CLS_POS_WEIGHT"]),
        decision_threshold=float(os.environ["DECISION_THRESHOLD"]),
    )

    model.train_model(
        job_id=job_id,
        lr=float(os.environ["LEARNING_RATE"]),
        num_epochs=int(os.environ["NUM_EPOCHS"]),
        log_interval=int(os.environ["LOG_INTERVAL"]),
        weight_decay=float(os.environ["WEIGHT_DECAY"]),
        lr_scheduler=lr_scheduler,
        lr_scheduler_factor=float(os.environ["LR_SCHEDULER_FACTOR"]),
        lr_scheduler_patience=int(os.environ["LR_SCHEDULER_PATIENCE"]),
        early_stopping_patience=int(os.environ["EARLY_STOPPING_PATIENCE"]),
        gradient_clip=float(os.environ["GRADIENT_CLIP"]),
        batch_size=int(os.environ["BATCH_SIZE"]),
        dataloader_num_workers=int(os.environ["DATALOADER_NUM_WORKERS"]),
        fbeta=float(os.environ.get("FBETA", "1.8")),
        warm_start_checkpoint=str(_abs(warm_start)) if warm_start else None,
    )


if __name__ == "__main__":
    main()
