"""
Data preprocessing script for DLDL project.

WARNING: Memory-intensive. For large datasets, use cpu_use=0.2-0.3 and run
operations one at a time.
"""

import os
import logging
from constants import NORMALIZATION
from model.preprocessor import Preprocessor
from util.processing import (
    get_processed_dataset_path,
    get_processed_labels_path,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration - shared with training workflow via constants.NORMALIZATION
DATASET_ID = NORMALIZATION

# Delete cached/processed files before preprocessing
logger.info("Cleaning up cached preprocessed files...")
dataset_path = get_processed_dataset_path(DATASET_ID)
labels_path = get_processed_labels_path(DATASET_ID)


if os.path.exists(dataset_path):
    os.remove(dataset_path)
    logger.info(f"Deleted cached dataset: {dataset_path}")
if os.path.exists(labels_path):
    os.remove(labels_path)
    logger.info(f"Deleted cached labels: {labels_path}")

# Create preprocessor (will automatically create dataset since files were deleted)
preprocessor = Preprocessor(
    dataset_id=DATASET_ID,
    cpu_use=0.2,
    normalization=NORMALIZATION,
)

# Verify dataset integrity and convert to float32
preprocessor.check_dataset(scale_labels=True)
preprocessor.convert_to_float()
