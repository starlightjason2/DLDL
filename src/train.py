"""Training entry point for DLDL disruption prediction model."""

import os
import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level="INFO",
)

from constants import (
    DATASET_DIR,
    JOB_ID,
    NORMALIZATION_TYPE,
    PMI_RANK,
    PMI_SIZE,
    PROG_DIR,
)
from model.cnn import IpCNN

if __name__ == "__main__":
    suffix = f"_{NORMALIZATION_TYPE}" if NORMALIZATION_TYPE else ""
    model = IpCNN(
        data_path=os.path.join(DATASET_DIR, f"processed_dataset{suffix}.pt"),
        labels_path=os.path.join(DATASET_DIR, f"processed_labels{suffix}.pt"),
        prog_dir=PROG_DIR,
        classification=False,
    )
    model.train_model(
        rank=int(PMI_RANK) if PMI_RANK is not None else 0,
        world_size=int(PMI_SIZE) if PMI_SIZE is not None else 1,
        job_id=JOB_ID,
        normalization_type=NORMALIZATION_TYPE,
        lr=0.0005,
        num_epochs=200,
        log_interval=50,
    )
