"""Data processing utilities for DLDL project."""

import multiprocessing as mp
import os
from typing import Tuple, cast
from collections.abc import Sized
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset, Subset


def get_use_cores(cpu_use: float) -> int:
    """Calculate number of CPU cores to use for parallel processing."""
    return max(1, int(cpu_use * mp.cpu_count()))


def create_binary_labels(shot_list: NDArray) -> NDArray[np.float64]:
    """Create binary classification labels from shotlist."""
    labels = np.copy(shot_list)
    labels[:, 0] = (shot_list[:, 1] != -1.0).astype(float)
    return labels


def convert_tensors_to_float(dset_path: str, labels_path: str) -> None:
    """Convert dataset and labels to float32."""
    dataset, labels = torch.load(dset_path), torch.load(labels_path)
    torch.save(dataset.float(), dset_path)
    torch.save(labels.float(), labels_path)


def split(dataset: Dataset, train_size: float = 0.8) -> Tuple[Subset, Subset, Subset]:
    """Split dataset into train, dev, and test sets."""
    total_size = len(cast(Sized, dataset))
    train_end = int(train_size * total_size)
    dev_end = int((train_size + (1 - train_size) / 2) * total_size)
    return (
        Subset(dataset, range(0, train_end)),
        Subset(dataset, range(train_end, dev_end)),
        Subset(dataset, range(dev_end, total_size)),
    )
