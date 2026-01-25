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
    "PROG_DIR",
    "JOB_ID",
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
PROG_DIR = _resolve_path(os.environ["PROG_DIR"])

# Ensure all required directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(PROG_DIR, exist_ok=True)

JOB_ID = os.environ["JOB_ID"]

import torch.nn as nn

CLASSIFICATION_LOSS: "nn.BCEWithLogitsLoss" = nn.BCEWithLogitsLoss()
TIME_PREDICTION_LOSS: "nn.MSELoss" = nn.MSELoss()

# Optional: Distributed training variables (set by job scheduler)
LOCAL_RANK = os.environ.get("PMI_LOCAL_RANK")
PMI_RANK = os.environ.get("PMI_RANK")
PMI_SIZE = os.environ.get("PMI_SIZE")


################################################################################
## Dataset Configuration
################################################################################
# Shared normalization configuration for preprocessing and training workflows
# This value is used as both the normalization method and dataset_id suffix
# Change this to switch between different preprocessing configs
NORMALIZATION = "meanvar-whole"
