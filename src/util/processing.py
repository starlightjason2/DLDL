"""Data processing utilities for DLDL project."""

import multiprocessing as mp
import os
from typing import Tuple, cast
from collections.abc import Sized
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset, Subset
from constants import DATASET_DIR


def get_use_cores(cpu_use: float) -> int:
    """Calculate number of CPU cores to use for parallel processing."""
    return max(1, int(cpu_use * mp.cpu_count()))


def create_binary_labels(shot_list: NDArray) -> NDArray[np.float64]:
    """Create binary classification labels from shotlist."""
    labels = np.copy(shot_list)
    labels[:, 0] = (shot_list[:, 1] != -1.0).astype(float)
    return labels


def convert_tensors_to_float(dset_path: str, labels_path: str) -> None:
    """Convert dataset and labels tensors to float32."""
    dataset = torch.load(dset_path)
    labels = torch.load(labels_path)
    torch.save(dataset.float(), dset_path)
    torch.save(labels.float(), labels_path)


def get_processed_dataset_path(dataset_id: str = "") -> str:
    """Get path to processed dataset file."""
    suffix = f"_{dataset_id}" if dataset_id else ""
    return os.path.join(DATASET_DIR, f"processed_dataset{suffix}.pt")


def get_processed_labels_path(dataset_id: str = "") -> str:
    """Get path to processed labels file."""
    suffix = f"_{dataset_id}" if dataset_id else ""
    return os.path.join(DATASET_DIR, f"processed_labels{suffix}.pt")


def split(dataset: Dataset, train_size: float = 0.8) -> Tuple[Subset, Subset, Subset]:
    """Split dataset into train, dev, and test sets. Returns (train, dev, test)."""
    dev_size: float = (1 - train_size) / 2
    total_size: int = len(cast(Sized, dataset))
    train_end: int = int(train_size * total_size)
    dev_end: int = int((train_size + dev_size) * total_size)
    train_indices = range(0, train_end)
    dev_indices = range(train_end, dev_end)
    test_indices = range(dev_end, total_size)

    train: Subset = Subset(dataset, train_indices)
    dev: Subset = Subset(dataset, dev_indices)
    test: Subset = Subset(dataset, test_indices)

    return train, dev, test
