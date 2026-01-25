import os
import numpy as np
import torch
from model import train
from constants import DATA_PATH, TRAIN_LABELS_PATH, MAX_LENGTH_FILE, PROG_DIR, JOB_ID


if __name__ == "__main__":
    max_length: int = int(np.loadtxt(MAX_LENGTH_FILE).astype(int))

    rank = int(os.getenv("PMI_RANK", "0"))
    world_size = int(os.getenv("PMI_SIZE", "1"))  # Default to 1 if not set
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
