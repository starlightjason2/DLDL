"""Training entry point for DLDL disruption prediction model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

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
    from model.cnn import IpCNN

    prog_dir = _abs(os.environ["PROG_DIR"])
    job_id = os.environ["JOB_ID"]
    data_path = _abs(os.environ["DATA_PATH"])
    labels_path = _abs(os.environ["TRAIN_LABELS_PATH"])

    # Ensure output directories exist before logging starts
    prog_dir.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    _configure_logging(prog_dir, job_id)

    e = os.environ
    model = IpCNN(
        data_path=str(data_path),
        labels_path=str(labels_path),
        prog_dir=str(prog_dir),
        conv1=(
            int(e["CONV1_FILTERS"]),
            int(e["CONV1_KERNEL"]),
            int(e["CONV1_PADDING"]),
        ),
        conv2=(
            int(e["CONV2_FILTERS"]),
            int(e["CONV2_KERNEL"]),
            int(e["CONV2_PADDING"]),
        ),
        conv3=(
            int(e["CONV3_FILTERS"]),
            int(e["CONV3_KERNEL"]),
            int(e["CONV3_PADDING"]),
        ),
        pool_size=int(e["POOL_SIZE"]),
        fc1_size=int(e["FC1_SIZE"]),
        fc2_size=int(e["FC2_SIZE"]),
        dropout_rate=float(e["DROPOUT_RATE"]),
        classification=False,
        normalization_type=e["NORMALIZATION_TYPE"],
    )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    model.train_model(
        rank=rank, world_size=world_size, local_rank=local_rank, job_id=job_id
    )


if __name__ == "__main__":
    main()
