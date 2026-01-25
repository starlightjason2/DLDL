"""Preprocessing utilities for DLDL project."""

import multiprocessing as mp
import numpy as np
from numpy.typing import NDArray

try:
    import torch
except ImportError:
    pass


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
