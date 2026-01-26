"""Data preprocessing script for DLDL."""

import os
import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", colorize=True, level="INFO")

import os
from constants import DATASET_DIR, NORMALIZATION_TYPE
from model.dataset import IpDataset

if __name__ == "__main__":
    logger.info("Cleaning up cached preprocessed files...")
    suffix = f"_{NORMALIZATION_TYPE}" if NORMALIZATION_TYPE else ""
    for path in (os.path.join(DATASET_DIR, f"processed_dataset{suffix}.pt"), os.path.join(DATASET_DIR, f"processed_labels{suffix}.pt")):
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted cached file: {path}")
    IpDataset().check_dataset(scale_labels=True)
