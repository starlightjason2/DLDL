"""Main entry point for training DLDL disruption prediction model."""

import sys

import torch
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level="INFO",
)

from constants import JOB_ID, NORMALIZATION_TYPE, PMI_RANK, PMI_SIZE, PROG_DIR
from model.model import IpCNN
from util.processing import get_processed_dataset_path, get_processed_labels_path

if __name__ == "__main__":
    data_path = get_processed_dataset_path(NORMALIZATION_TYPE)
    labels_path = get_processed_labels_path(NORMALIZATION_TYPE)

    # Load dataset to get max_length (will be loaded again in train_model for validation)
    dataset = torch.load(data_path)
    max_length = int(dataset.shape[1])
    del dataset  # Free memory before training

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
        dataset_id=NORMALIZATION_TYPE,
        lr=0.0005,
        num_epochs=250,
        log_interval=50,
    )
