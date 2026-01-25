"""Data loading utilities for plasma disruption datasets."""

import os
from typing import Tuple, List, Optional
import numpy as np
from numpy.typing import NDArray


def get_length(filename: str, data_dir: str) -> int:
    """Get time series length for a single file."""
    return len(np.loadtxt(os.path.join(data_dir, filename), usecols=1))


def get_scaled_t_disrupt(
    shot_no: int, data_dir: str, t_disrupt: float, max_length: int
) -> float:
    """Compute normalized disruption time index [0, 1]."""
    if max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {max_length}")
    time = np.loadtxt(os.path.join(data_dir, f"{shot_no}.txt"), usecols=0)
    return int(np.abs(time - t_disrupt).argmin()) / max_length


def get_means(filename: str, data_dir: str) -> List[float]:
    """Compute mean and mean of squares. Returns [mean, mean_squared] for std calculation."""
    data = np.loadtxt(os.path.join(data_dir, filename), usecols=1)
    return [float(np.mean(data)), float(np.mean(data**2))]


def load_and_pad(
    filename: str, data_dir: str, max_length: int
) -> Tuple[int, NDArray[np.float32]]:
    """Load time series and pad with zeros to max_length. Returns (shot_number, padded_data)."""
    shot_no = int(filename[:-4])
    data = np.loadtxt(os.path.join(data_dir, filename), usecols=1, dtype=np.float32)
    padded = np.zeros(max_length, dtype=np.float32)
    length = min(len(data), max_length)
    padded[:length] = data[:length]
    return shot_no, padded


def load_and_pad_norm(
    filename: str,
    data_dir: str,
    max_length: int,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[int, NDArray[np.float32]]:
    """Load, Z-score normalize, and pad. If mean/std None, computes per-shot. Returns (shot_number, normalized_padded_data)."""
    shot_no = int(filename[:-4])
    data = np.loadtxt(os.path.join(data_dir, filename), usecols=1, dtype=np.float32)

    if mean is None or std is None:
        mean, std = float(np.mean(data)), float(np.std(data))

    if std > 0:
        data = (data - mean) / std
    else:
        data = np.zeros_like(data)  # Constant signal, set to zero

    padded = np.zeros(max_length, dtype=np.float32)
    length = min(len(data), max_length)
    padded[:length] = data[:length]
    return shot_no, padded


def load_and_pad_scale(
    filename: str, data_dir: str, max_length: int
) -> Tuple[int, NDArray[np.float32]]:
    """Load, min-max scale to [0,1], and pad. Returns (shot_number, scaled_padded_data)."""
    shot_no = int(filename[:-4])
    data = np.loadtxt(os.path.join(data_dir, filename), usecols=1, dtype=np.float32)

    data_min, data_max = np.min(data), np.max(data)
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)
    else:
        data = np.zeros_like(data)  # Constant signal, set to zero

    padded = np.zeros(max_length, dtype=np.float32)
    length = min(len(data), max_length)
    padded[:length] = data[:length]
    return shot_no, padded


def check_file(file_path: str, verbose: bool = False) -> bool:
    """Check if file exists. If verbose, print file size or non-existence message."""
    if os.path.exists(file_path):
        if verbose:
            print(f"File {file_path} exists. Size: {os.path.getsize(file_path)} bytes.")
        return True
    if verbose:
        print(f"File {file_path} does not exist.")
    return False
