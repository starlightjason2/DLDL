"""Validation script: runs preprocessed file checks and dataset integrity verification."""

import argparse
import os
import sys

from loguru import logger

from constants import DATASET_DIR, NORMALIZATION_TYPE, PROG_DIR

# Configure logging
logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
logger.add(sys.stderr, format=log_format, colorize=True, level="INFO")
logger.add(
    os.path.join(PROG_DIR, "validate.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
)

from model.dataset import IpDataset


def validate_preprocessed_files() -> None:
    """Validate that preprocessed files exist (IpCNN.validate_preprocessed_files)."""
    suffix = f"_{NORMALIZATION_TYPE}" if NORMALIZATION_TYPE else ""
    data_path = os.path.join(DATASET_DIR, f"processed_dataset{suffix}.pt")
    labels_path = os.path.join(DATASET_DIR, f"processed_labels{suffix}.pt")

    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        missing = [
            f"Dataset: {data_path}" if not os.path.exists(data_path) else None,
            f"Labels: {labels_path}" if not os.path.exists(labels_path) else None,
        ]
        missing = [m for m in missing if m]
        logger.error(
            f"Preprocessed files not found for NORMALIZATION_TYPE='{NORMALIZATION_TYPE}'. "
            f"Missing: {', '.join(missing)}"
        )
        raise FileNotFoundError(
            "Preprocessed files not found. Run preprocess_data.py first."
        )
    logger.info("Preprocessed files exist: OK")


def check_dataset(num_checks: int = 100, scale_labels: bool = True, verbose: bool = False) -> None:
    """Run dataset integrity check (IpDataset.check_dataset)."""
    dataset = IpDataset(normalization_type=NORMALIZATION_TYPE)
    dataset.check_dataset(scale_labels=scale_labels, num_checks=num_checks, verbose=verbose)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate preprocessed data and dataset integrity")
    parser.add_argument(
        "--num-checks",
        type=int,
        default=100,
        help="Number of examples to verify in integrity check (default: 100)",
    )
    parser.add_argument(
        "--no-scale-labels",
        action="store_true",
        help="Disable label scaling when verifying (default: scale_labels=True)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output during integrity check",
    )
    parser.add_argument(
        "--skip-integrity",
        action="store_true",
        help="Only check that preprocessed files exist; skip integrity verification",
    )
    args = parser.parse_args()

    logger.info("Running validation...")
    validate_preprocessed_files()

    if not args.skip_integrity:
        check_dataset(
            num_checks=args.num_checks,
            scale_labels=not args.no_scale_labels,
            verbose=args.verbose,
        )

    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
