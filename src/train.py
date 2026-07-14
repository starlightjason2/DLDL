"""Training entry point for DLDL disruption prediction model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from model.dataset import IpDataset
from util.training import build_cnn_from_env

_REPO = Path(__file__).resolve().parents[1]  # project root (src/..)
# Env paths are relative to the repo root; run from there so they resolve directly.
os.chdir(_REPO)
load_dotenv(dotenv_path=_REPO / ".env", encoding="utf-8")


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
    prog_dir = Path(os.environ["PROG_DIR"])
    job_id = os.environ["JOB_ID"]
    data_path = Path(os.environ["DATA_PATH"])
    labels_path = Path(os.environ["TRAIN_LABELS_PATH"])

    # Ensure output directories exist before logging starts
    prog_dir.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    _configure_logging(prog_dir, job_id)

    lr_scheduler = os.environ["LR_SCHEDULER"].lower() in ("true", "1", "yes", "on")

    dataset = IpDataset(
        
        data_file=str(data_path),
        labels_file=str(labels_path),
        labels_path=os.environ["LABELS_PATH"],
        data_dir=os.environ["DATA_DIR"],
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )

    model = build_cnn_from_env(dataset, str(prog_dir))

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
    )


if __name__ == "__main__":
    main()
