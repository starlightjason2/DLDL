"""Data preprocessing script for DLDL."""

import os
import sys
from loguru import logger

from constants import DATASET_DIR, NORMALIZATION_TYPE, PROG_DIR

# Configure logging: stderr + file
logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
logger.add(sys.stderr, format=log_format, colorize=True, level="INFO")
logger.add(
    os.path.join(PROG_DIR, "preprocess.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
)
from model.dataset import IpDataset

if __name__ == "__main__":
    logger.info("Cleaning up cached preprocessed files...")
    suffix = f"_{NORMALIZATION_TYPE}" if NORMALIZATION_TYPE else ""
    for path in (os.path.join(DATASET_DIR, f"processed_dataset{suffix}.pt"), os.path.join(DATASET_DIR, f"processed_labels{suffix}.pt")):
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted cached file: {path}")
    IpDataset(normalization_type=NORMALIZATION_TYPE).check_dataset(scale_labels=True)
