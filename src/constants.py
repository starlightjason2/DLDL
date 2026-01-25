"""
Project-wide constants for DLDL disruption prediction system.

All path constants loaded from environment variables. All env vars are REQUIRED.
See .env.local.example or .env.polaris.example for templates.
"""

import os

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
## Dataset Configuration (optional env; defaults below)
################################################################################
# NORMALIZATION_TYPE: method + dataset_id suffix. Options: scale, meanvar-whole, meanvar-single.
# CPU_USE: fraction of CPU cores for preprocessing (0-1). Use 0.2-0.3 for ~32GB RAM.
_NORMALIZATION_TYPE_RAW = os.environ.get("NORMALIZATION_TYPE", "meanvar-whole")
_CPU_USE_RAW = os.environ.get("CPU_USE", "0.2")

NORMALIZATION_TYPE: str = _NORMALIZATION_TYPE_RAW
try:
    CPU_USE: float = float(_CPU_USE_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"CPU_USE must be a number, got {_CPU_USE_RAW!r}") from e

if not (0 < CPU_USE <= 1):
    raise ValueError(f"CPU_USE must be in (0, 1], got {CPU_USE}")
if NORMALIZATION_TYPE not in ("scale", "meanvar-whole", "meanvar-single"):
    raise ValueError(
        f"NORMALIZATION_TYPE must be 'scale', 'meanvar-whole', or 'meanvar-single'; got {NORMALIZATION_TYPE!r}"
    )
