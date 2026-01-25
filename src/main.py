"""Main entry point for training DLDL disruption prediction model."""

import os
import torch
from model import train
from constants import DATA_PATH, TRAIN_LABELS_PATH, PROG_DIR, JOB_ID


if __name__ == "__main__":
    dataset = torch.load(DATA_PATH)
    max_length = int(dataset.shape[1])

    rank = int(os.getenv("PMI_RANK", "0"))
    world_size = int(os.getenv("PMI_SIZE", "1"))

    print("GPUs Available:", torch.cuda.device_count())
    print("Rank:", rank)

    train(
        rank,
        world_size,
        DATA_PATH,
        TRAIN_LABELS_PATH,
        PROG_DIR,
        max_length,
        job_id=JOB_ID,
        lr=0.0005,
        num_epochs=250,
        log_interval=50,
    )
