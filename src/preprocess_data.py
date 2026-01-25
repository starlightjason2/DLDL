"""Data preprocessing script for DLDL. Set CPU_USE=0.2-0.3 for ~32GB RAM."""

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

from constants import NORMALIZATION_TYPE
from model.preprocessor import Preprocessor
from util.processing import get_processed_dataset_path, get_processed_labels_path

if __name__ == "__main__":
    logger.info("Cleaning up cached preprocessed files...")
    for path in (
        get_processed_dataset_path(NORMALIZATION_TYPE),
        get_processed_labels_path(NORMALIZATION_TYPE),
    ):
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted cached file: {path}")

    preprocessor = Preprocessor()
    preprocessor.check_dataset(scale_labels=True)
