"""Training entry point for DLDL disruption prediction model."""

import os
import sys
from loguru import logger

from constants import (
    DATASET_DIR,
    JOB_ID,
    NORMALIZATION_TYPE,
    PMI_RANK,
    PMI_SIZE,
    PROG_DIR,
    CONV1_FILTERS,
    CONV1_KERNEL,
    CONV1_PADDING,
    CONV2_FILTERS,
    CONV2_KERNEL,
    CONV2_PADDING,
    CONV3_FILTERS,
    CONV3_KERNEL,
    CONV3_PADDING,
    POOL_SIZE,
    FC1_SIZE,
    FC2_SIZE,
    DROPOUT_RATE,
)

# Configure logging: stderr + file
logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
logger.add(sys.stderr, format=log_format, colorize=True, level="INFO")
logger.add(
    os.path.join(PROG_DIR, f"{JOB_ID}.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
)

from model.cnn import IpCNN

if __name__ == "__main__":
    suffix = f"_{NORMALIZATION_TYPE}" if NORMALIZATION_TYPE else ""
    model = IpCNN(
        data_path=os.path.join(DATASET_DIR, f"processed_dataset{suffix}.pt"),
        labels_path=os.path.join(DATASET_DIR, f"processed_labels{suffix}.pt"),
        prog_dir=PROG_DIR,
        conv1=(CONV1_FILTERS, CONV1_KERNEL, CONV1_PADDING),
        conv2=(CONV2_FILTERS, CONV2_KERNEL, CONV2_PADDING),
        conv3=(CONV3_FILTERS, CONV3_KERNEL, CONV3_PADDING),
        pool_size=POOL_SIZE,
        fc1_size=FC1_SIZE,
        fc2_size=FC2_SIZE,
        dropout_rate=DROPOUT_RATE,
        classification=False,
        normalization_type=NORMALIZATION_TYPE,
    )
    model.train_model(
        rank=int(PMI_RANK) if PMI_RANK is not None else 0,
        world_size=int(PMI_SIZE) if PMI_SIZE is not None else 1,
        job_id=JOB_ID,
    )
