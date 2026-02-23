"""Project-wide constants for DLDL disruption prediction system."""

import os

import torch.nn as nn
from dotenv import load_dotenv

# Get project root directory (one level up from src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load .env file if available
env_file = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(env_file):
    load_dotenv(env_file)

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


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


def _validate_constants() -> None:
    if not (0 < CPU_USE <= 1):
        raise ValueError(f"CPU_USE must be in (0, 1], got {CPU_USE}")
    if NORMALIZATION_TYPE not in ("scale", "meanvar-whole", "meanvar-single"):
        raise ValueError(f"NORMALIZATION_TYPE must be 'scale', 'meanvar-whole', or 'meanvar-single'; got {NORMALIZATION_TYPE!r}")


missing_vars = [var for var in _REQUIRED_ENV_VARS if var not in os.environ]
if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

DATA_DIR = _resolve_path(os.environ["DATA_DIR"])
DATASET_DIR = _resolve_path(os.environ["DATASET_DIR"])
LABELS_PATH = _resolve_path(os.environ["LABELS_PATH"])
PROG_DIR = _resolve_path(os.environ["PROG_DIR"])

# Ensure all required directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(PROG_DIR, exist_ok=True)

JOB_ID = os.environ["JOB_ID"]

# Optional: Graph output directory (defaults to PROG_DIR if not set)
if "GRAPH_DIR" in os.environ:
    GRAPH_DIR = _resolve_path(os.environ["GRAPH_DIR"])
else:
    GRAPH_DIR = PROG_DIR
os.makedirs(GRAPH_DIR, exist_ok=True)

CLASSIFICATION_LOSS = nn.BCEWithLogitsLoss()
TIME_PREDICTION_LOSS = nn.MSELoss()

# Optional: Distributed training variables (set by job scheduler)
LOCAL_RANK = os.environ.get("PMI_LOCAL_RANK")
PMI_RANK = os.environ.get("PMI_RANK")
PMI_SIZE = os.environ.get("PMI_SIZE")

# Optional: Training configuration
_EARLY_STOPPING_PATIENCE_RAW = os.environ.get("EARLY_STOPPING_PATIENCE", "10")
try:
    EARLY_STOPPING_PATIENCE = int(_EARLY_STOPPING_PATIENCE_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"EARLY_STOPPING_PATIENCE must be an integer, got {_EARLY_STOPPING_PATIENCE_RAW!r}") from e

# Hyperparameters
_LEARNING_RATE_RAW = os.environ.get("LEARNING_RATE", "0.0005")
try:
    LEARNING_RATE = float(_LEARNING_RATE_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"LEARNING_RATE must be a number, got {_LEARNING_RATE_RAW!r}") from e

_NUM_EPOCHS_RAW = os.environ.get("NUM_EPOCHS", "200")
try:
    NUM_EPOCHS = int(_NUM_EPOCHS_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"NUM_EPOCHS must be an integer, got {_NUM_EPOCHS_RAW!r}") from e

_LOG_INTERVAL_RAW = os.environ.get("LOG_INTERVAL", "50")
try:
    LOG_INTERVAL = int(_LOG_INTERVAL_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"LOG_INTERVAL must be an integer, got {_LOG_INTERVAL_RAW!r}") from e

_WEIGHT_DECAY_RAW = os.environ.get("WEIGHT_DECAY", "0.0001")
try:
    WEIGHT_DECAY = float(_WEIGHT_DECAY_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"WEIGHT_DECAY must be a number, got {_WEIGHT_DECAY_RAW!r}") from e

_DROPOUT_RATE_RAW = os.environ.get("DROPOUT_RATE", "0.35")
try:
    DROPOUT_RATE = float(_DROPOUT_RATE_RAW)
    if not (0 <= DROPOUT_RATE <= 1):
        raise ValueError(f"DROPOUT_RATE must be in [0, 1], got {DROPOUT_RATE}")
except (TypeError, ValueError) as e:
    raise ValueError(f"DROPOUT_RATE must be a number in [0, 1], got {_DROPOUT_RATE_RAW!r}") from e

_BATCH_SIZE_RAW = os.environ.get("BATCH_SIZE", "128")
try:
    BATCH_SIZE = int(_BATCH_SIZE_RAW)
    if BATCH_SIZE <= 0:
        raise ValueError(f"BATCH_SIZE must be > 0, got {BATCH_SIZE}")
except (TypeError, ValueError) as e:
    raise ValueError(f"BATCH_SIZE must be a positive integer, got {_BATCH_SIZE_RAW!r}") from e

_DATALOADER_NUM_WORKERS_RAW = os.environ.get("DATALOADER_NUM_WORKERS", "4")
try:
    DATALOADER_NUM_WORKERS = int(_DATALOADER_NUM_WORKERS_RAW)
    if DATALOADER_NUM_WORKERS < 0:
        raise ValueError(f"DATALOADER_NUM_WORKERS must be >= 0, got {DATALOADER_NUM_WORKERS}")
except (TypeError, ValueError) as e:
    raise ValueError(f"DATALOADER_NUM_WORKERS must be a non-negative integer, got {_DATALOADER_NUM_WORKERS_RAW!r}") from e

_LR_SCHEDULER_RAW = os.environ.get("LR_SCHEDULER", "true")
LR_SCHEDULER = _LR_SCHEDULER_RAW.lower() in ("true", "1", "yes", "on")

_LR_SCHEDULER_FACTOR_RAW = os.environ.get("LR_SCHEDULER_FACTOR", "0.5")
try:
    LR_SCHEDULER_FACTOR = float(_LR_SCHEDULER_FACTOR_RAW)
    if not (0 < LR_SCHEDULER_FACTOR < 1):
        raise ValueError(f"LR_SCHEDULER_FACTOR must be in (0, 1), got {LR_SCHEDULER_FACTOR}")
except (TypeError, ValueError) as e:
    raise ValueError(f"LR_SCHEDULER_FACTOR must be a number in (0, 1), got {_LR_SCHEDULER_FACTOR_RAW!r}") from e

_LR_SCHEDULER_PATIENCE_RAW = os.environ.get("LR_SCHEDULER_PATIENCE", "5")
try:
    LR_SCHEDULER_PATIENCE = int(_LR_SCHEDULER_PATIENCE_RAW)
    if LR_SCHEDULER_PATIENCE <= 0:
        raise ValueError(f"LR_SCHEDULER_PATIENCE must be > 0, got {LR_SCHEDULER_PATIENCE}")
except (TypeError, ValueError) as e:
    raise ValueError(f"LR_SCHEDULER_PATIENCE must be a positive integer, got {_LR_SCHEDULER_PATIENCE_RAW!r}") from e

_GRADIENT_CLIP_RAW = os.environ.get("GRADIENT_CLIP", "1.0")
try:
    GRADIENT_CLIP = float(_GRADIENT_CLIP_RAW)
    if GRADIENT_CLIP < 0:
        raise ValueError(f"GRADIENT_CLIP must be >= 0, got {GRADIENT_CLIP}")
except (TypeError, ValueError) as e:
    raise ValueError(f"GRADIENT_CLIP must be a non-negative number, got {_GRADIENT_CLIP_RAW!r}") from e

# Architecture hyperparameters
_CONV1_FILTERS_RAW = os.environ.get("CONV1_FILTERS", "16")
try:
    CONV1_FILTERS = int(_CONV1_FILTERS_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"CONV1_FILTERS must be an integer, got {_CONV1_FILTERS_RAW!r}") from e

_CONV1_KERNEL_RAW = os.environ.get("CONV1_KERNEL", "9")
try:
    CONV1_KERNEL = int(_CONV1_KERNEL_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"CONV1_KERNEL must be an integer, got {_CONV1_KERNEL_RAW!r}") from e

_CONV1_PADDING_RAW = os.environ.get("CONV1_PADDING", "4")
try:
    CONV1_PADDING = int(_CONV1_PADDING_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"CONV1_PADDING must be an integer, got {_CONV1_PADDING_RAW!r}") from e

_CONV2_FILTERS_RAW = os.environ.get("CONV2_FILTERS", "32")
try:
    CONV2_FILTERS = int(_CONV2_FILTERS_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"CONV2_FILTERS must be an integer, got {_CONV2_FILTERS_RAW!r}") from e

_CONV2_KERNEL_RAW = os.environ.get("CONV2_KERNEL", "5")
try:
    CONV2_KERNEL = int(_CONV2_KERNEL_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"CONV2_KERNEL must be an integer, got {_CONV2_KERNEL_RAW!r}") from e

_CONV2_PADDING_RAW = os.environ.get("CONV2_PADDING", "2")
try:
    CONV2_PADDING = int(_CONV2_PADDING_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"CONV2_PADDING must be an integer, got {_CONV2_PADDING_RAW!r}") from e

_CONV3_FILTERS_RAW = os.environ.get("CONV3_FILTERS", "64")
try:
    CONV3_FILTERS = int(_CONV3_FILTERS_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"CONV3_FILTERS must be an integer, got {_CONV3_FILTERS_RAW!r}") from e

_CONV3_KERNEL_RAW = os.environ.get("CONV3_KERNEL", "3")
try:
    CONV3_KERNEL = int(_CONV3_KERNEL_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"CONV3_KERNEL must be an integer, got {_CONV3_KERNEL_RAW!r}") from e

_CONV3_PADDING_RAW = os.environ.get("CONV3_PADDING", "1")
try:
    CONV3_PADDING = int(_CONV3_PADDING_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"CONV3_PADDING must be an integer, got {_CONV3_PADDING_RAW!r}") from e

_POOL_SIZE_RAW = os.environ.get("POOL_SIZE", "4")
try:
    POOL_SIZE = int(_POOL_SIZE_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"POOL_SIZE must be an integer, got {_POOL_SIZE_RAW!r}") from e

_FC1_SIZE_RAW = os.environ.get("FC1_SIZE", "120")
try:
    FC1_SIZE = int(_FC1_SIZE_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"FC1_SIZE must be an integer, got {_FC1_SIZE_RAW!r}") from e

_FC2_SIZE_RAW = os.environ.get("FC2_SIZE", "60")
try:
    FC2_SIZE = int(_FC2_SIZE_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"FC2_SIZE must be an integer, got {_FC2_SIZE_RAW!r}") from e


################################################################################
## Dataset Configuration (optional env; defaults below)
################################################################################
# NORMALIZATION_TYPE: method + filename suffix. Options: scale, meanvar-whole, meanvar-single.
# CPU_USE: fraction of CPU cores for preprocessing (0-1). Use 0.2-0.3 for ~32GB RAM.
_NORMALIZATION_TYPE_RAW = os.environ.get("NORMALIZATION_TYPE", "meanvar-whole")
_CPU_USE_RAW = os.environ.get("CPU_USE", "0.2")

NORMALIZATION_TYPE = _NORMALIZATION_TYPE_RAW
try:
    CPU_USE = float(_CPU_USE_RAW)
except (TypeError, ValueError) as e:
    raise ValueError(f"CPU_USE must be a number, got {_CPU_USE_RAW!r}") from e

# PREPROCESSOR_MAX_WORKERS: hard cap on ProcessPoolExecutor workers (avoids fork/resource issues on HPC)
_PREPROCESSOR_MAX_WORKERS_RAW = os.environ.get("PREPROCESSOR_MAX_WORKERS", "4")
try:
    PREPROCESSOR_MAX_WORKERS = int(_PREPROCESSOR_MAX_WORKERS_RAW)
    if PREPROCESSOR_MAX_WORKERS < 1:
        raise ValueError(f"PREPROCESSOR_MAX_WORKERS must be >= 1, got {PREPROCESSOR_MAX_WORKERS}")
except (TypeError, ValueError) as e:
    raise ValueError(f"PREPROCESSOR_MAX_WORKERS must be a positive integer, got {_PREPROCESSOR_MAX_WORKERS_RAW!r}") from e

# Validate all constants
_validate_constants()
