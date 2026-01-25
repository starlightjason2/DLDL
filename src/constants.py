"""
Constants for DLDL project.
"""
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

################################################################################
## Path Constants
################################################################################
# Data directories
DATA_DIR = '/eagle/fusiondl_aesp/signal_data/d3d/ipspr15V/'
DATASET_DIR = '/eagle/fusiondl_aesp/jrodriguez/processed_data/'
LABELS_PATH = '/eagle/fusiondl_aesp/jrodriguez/shot_lists/ips_labels.txt'

# Training data paths
DATA_PATH = "/eagle/fusiondl_aesp/jrodriguez/processed_data/processed_dataset_meanvar-whole.pt"
TRAIN_LABELS_PATH = "/eagle/fusiondl_aesp/jrodriguez/processed_data/processed_labels_scaled_labels.pt"
MAX_LENGTH_FILE = "/eagle/fusiondl_aesp/jrodriguez/processed_data/max_length.txt"
PROG_DIR = "/eagle/fusiondl_aesp/jrodriguez/train_progress/"

################################################################################
## Training Constants
################################################################################
JOB_ID = "DLDL_test_lr0005"

################################################################################
## Loss Function Constants
################################################################################
try:
    import torch.nn as nn
    CLASSIFICATION_LOSS: "nn.BCEWithLogitsLoss" = nn.BCEWithLogitsLoss()
    TIME_PREDICTION_LOSS: "nn.MSELoss" = nn.MSELoss()
except:
    # If torch is not installed, set to None
    CLASSIFICATION_LOSS: Any = None
    TIME_PREDICTION_LOSS: Any = None

################################################################################
## Environment Variable Constants
################################################################################
LOCAL_RANK = os.environ.get("PMI_LOCAL_RANK")
