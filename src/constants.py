"""
Project-wide constants for DLDL disruption prediction system.

This module centralizes all configuration constants including file paths,
training parameters, and loss function instances. All path constants are
loaded from environment variables, allowing easy switching between different
environments (local, Polaris, etc.) via .env files.

Usage:
    1. Copy .env.local.example to .env.local (for local development) or
       .env.polaris.example to .env.polaris (for Polaris HPC)
    2. Modify the paths in your .env file as needed
    3. Create a symlink: ln -s .env.local .env (or .env.polaris)
    4. The constants will automatically load from the .env file if python-dotenv
       is installed, otherwise set environment variables manually

Environment Variables:
    All constants use the DLDL_ prefix:
    - DLDL_DATA_DIR: Raw signal data directory
    - DLDL_DATASET_DIR: Preprocessed data directory
    - DLDL_LABELS_PATH: Path to labels file
    - DLDL_DATA_PATH: Preprocessed dataset file
    - DLDL_TRAIN_LABELS_PATH: Preprocessed labels file
    - DLDL_MAX_LENGTH_FILE: Max length metadata file
    - DLDL_PROG_DIR: Training progress/output directory
    - DLDL_JOB_ID: Training run identifier
"""

import os
from typing import TYPE_CHECKING, Any

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    # Load .env file from project root (one level up from src/)
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)
except ImportError:
    # python-dotenv not installed, skip automatic .env loading
    # Environment variables must be set manually or via system
    pass

if TYPE_CHECKING:
    import torch.nn as nn

################################################################################
## Path Constants (from environment variables)
################################################################################
# Raw data directory containing individual shot files (one .txt file per shot)
DATA_DIR = os.environ.get("DLDL_DATA_DIR", "/data/signal_data/")

# Directory where preprocessed datasets and metadata are stored
DATASET_DIR = os.environ.get("DLDL_DATASET_DIR", "/data/processed_data/")

# Path to labels file containing shot numbers and disruption times
LABELS_PATH = os.environ.get("DLDL_LABELS_PATH", "/data/shot_lists/ips_labels.txt")

# Preprocessed dataset file for training
DATA_PATH = os.environ.get(
    "DLDL_DATA_PATH", "/data/processed_data/processed_dataset_meanvar-whole.pt"
)

# Preprocessed labels file for training
TRAIN_LABELS_PATH = os.environ.get(
    "DLDL_TRAIN_LABELS_PATH",
    "/data/processed_data/processed_labels_scaled_labels.pt",
)

# File containing the maximum sequence length across all shots
MAX_LENGTH_FILE = os.environ.get(
    "DLDL_MAX_LENGTH_FILE", "/data/processed_data/max_length.txt"
)

# Directory for saving training progress, checkpoints, and logs
PROG_DIR = os.environ.get("DLDL_PROG_DIR", "/data/train_progress/")

################################################################################
## Training Constants
################################################################################
# Unique identifier for this training run (used in filenames)
JOB_ID = os.environ.get("DLDL_JOB_ID", "DLDL_test_lr0005")

################################################################################
## Loss Function Constants
################################################################################
try:
    import torch.nn as nn

    # Binary cross-entropy loss for classification task
    CLASSIFICATION_LOSS: "nn.BCEWithLogitsLoss" = nn.BCEWithLogitsLoss()
    # Mean squared error loss for time prediction task
    TIME_PREDICTION_LOSS: "nn.MSELoss" = nn.MSELoss()
except ImportError:
    # If torch is not installed, set to None to allow type checking
    CLASSIFICATION_LOSS: Any = None
    TIME_PREDICTION_LOSS: Any = None

################################################################################
## Environment Variable Constants
################################################################################
# Local rank for distributed training (from PMI environment)
LOCAL_RANK = os.environ.get("PMI_LOCAL_RANK")
