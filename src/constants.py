"""
Project-wide constants for DLDL disruption prediction system.

All path constants loaded from environment variables. All env vars are REQUIRED.
See .env.local.example or .env.polaris.example for templates.
"""

import os
from typing import Any

# Get project root directory (one level up from src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(path: str) -> str:
    """Convert relative paths to absolute paths based on project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)


# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    env_file = os.path.join(_PROJECT_ROOT, ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)
except ImportError:
    pass

################################################################################
## Path Constants (from environment variables)
################################################################################
# Required environment variables
_REQUIRED_ENV_VARS = [
    "DATA_DIR",
    "DATASET_DIR",
    "LABELS_PATH",
    "DATA_PATH",
    "TRAIN_LABELS_PATH",
    "PROG_DIR",
    "JOB_ID",
    "SCALED_LABELS_FILENAME",
]


# Check that all required environment variables are set
_missing_vars = [var for var in _REQUIRED_ENV_VARS if var not in os.environ]
if _missing_vars:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(_missing_vars)}\n"
        f"Please set these variables or create a .env file. See .env.local.example "
        f"or .env.polaris.example for templates."
    )

DATA_DIR = _resolve_path(os.environ["DATA_DIR"])
DATASET_DIR = _resolve_path(os.environ["DATASET_DIR"])
LABELS_PATH = _resolve_path(os.environ["LABELS_PATH"])
DATA_PATH = _resolve_path(os.environ["DATA_PATH"])
TRAIN_LABELS_PATH = _resolve_path(os.environ["TRAIN_LABELS_PATH"])
SCALED_LABELS_FILENAME = os.environ["SCALED_LABELS_FILENAME"]
PROG_DIR = _resolve_path(os.environ["PROG_DIR"])

# Ensure all required directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(PROG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(TRAIN_LABELS_PATH), exist_ok=True)

JOB_ID = os.environ["JOB_ID"]

try:
    import torch.nn as nn

    CLASSIFICATION_LOSS: "nn.BCEWithLogitsLoss" = nn.BCEWithLogitsLoss()
    TIME_PREDICTION_LOSS: "nn.MSELoss" = nn.MSELoss()
except ImportError:
    CLASSIFICATION_LOSS: Any = None
    TIME_PREDICTION_LOSS: Any = None

# Optional: Distributed training variables (set by job scheduler)
LOCAL_RANK = os.environ.get("PMI_LOCAL_RANK")
PMI_RANK = os.environ.get("PMI_RANK")
PMI_SIZE = os.environ.get("PMI_SIZE")

################################################################################
## Path Helper Functions
################################################################################


def get_processed_dataset_path(dataset_id: str = "") -> str:
    """Get path to processed dataset file."""
    return os.path.join(DATASET_DIR, f"processed_dataset{dataset_id}.pt")


def get_processed_labels_path(dataset_id: str = "") -> str:
    """Get path to processed labels file."""
    return os.path.join(DATASET_DIR, f"processed_labels{dataset_id}.pt")
