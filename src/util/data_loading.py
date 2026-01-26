"""Data loading utilities for plasma disruption datasets."""

import os
from typing import Tuple, List, Optional
import numpy as np
from numpy.typing import NDArray


def get_length(filename: str, data_dir: str) -> int:
    return len(np.loadtxt(os.path.join(data_dir, filename), usecols=1))


def get_scaled_t_disrupt(shot_no: int, data_dir: str, t_disrupt: float, max_length: int) -> float:
    if max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {max_length}")
    time = np.loadtxt(os.path.join(data_dir, f"{shot_no}.txt"), usecols=0)
    return int(np.abs(time - t_disrupt).argmin()) / max_length


def get_means(filename: str, data_dir: str) -> List[float]:
    data = np.loadtxt(os.path.join(data_dir, filename), usecols=1)
    return [float(np.mean(data)), float(np.mean(data**2))]


def _load_and_pad_base(filename: str, data_dir: str, max_length: int, data: NDArray) -> Tuple[int, NDArray[np.float32]]:
    """Base function for loading and padding."""
    shot_no = int(filename[:-4])
    padded = np.zeros(max_length, dtype=np.float32)
    padded[:min(len(data), max_length)] = data[:min(len(data), max_length)]
    return shot_no, padded


def load_and_pad(filename: str, data_dir: str, max_length: int) -> Tuple[int, NDArray[np.float32]]:
    data = np.loadtxt(os.path.join(data_dir, filename), usecols=1, dtype=np.float32)
    return _load_and_pad_base(filename, data_dir, max_length, data)


def load_and_pad_norm(
    filename: str, data_dir: str, max_length: int, mean: Optional[float] = None, std: Optional[float] = None
) -> Tuple[int, NDArray[np.float32]]:
    data = np.loadtxt(os.path.join(data_dir, filename), usecols=1, dtype=np.float32)
    if mean is None or std is None:
        mean, std = float(np.mean(data)), float(np.std(data))
    data = (data - mean) / std if std > 0 else np.zeros_like(data)
    return _load_and_pad_base(filename, data_dir, max_length, data)


def load_and_pad_scale(filename: str, data_dir: str, max_length: int) -> Tuple[int, NDArray[np.float32]]:
    data = np.loadtxt(os.path.join(data_dir, filename), usecols=1, dtype=np.float32)
    data_min, data_max = np.min(data), np.max(data)
    data = (data - data_min) / (data_max - data_min) if data_max > data_min else np.zeros_like(data)
    return _load_and_pad_base(filename, data_dir, max_length, data)
