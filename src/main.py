"""Main entry point for training DLDL disruption prediction model."""

import logging
import torch
from constants import JOB_ID, NORMALIZATION, PMI_RANK, PMI_SIZE, PROG_DIR
from model.model import IpCNN
from util.processing import get_processed_dataset_path, get_processed_labels_path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration - shared with preprocessing workflow via constants.NORMALIZATION
DATASET_ID = NORMALIZATION

if __name__ == "__main__":
    data_path = get_processed_dataset_path(DATASET_ID)
    labels_path = get_processed_labels_path(DATASET_ID)

    dataset = torch.load(data_path)
    max_length = int(dataset.shape[1])

    rank = int(PMI_RANK) if PMI_RANK is not None else 0
    world_size = int(PMI_SIZE) if PMI_SIZE is not None else 1

    model = IpCNN(max_length=max_length, classification=False)
    model.train_model(
        rank,
        world_size,
        data_path,
        labels_path,
        PROG_DIR,
        JOB_ID,
        dataset_id=DATASET_ID,
        lr=0.0005,
        num_epochs=250,
        log_interval=50,
    )
