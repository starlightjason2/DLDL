"""Data loading utilities for plasma disruption datasets."""

import os
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _read_signal_file(filepath: str, col: int = 1) -> np.ndarray:
    """Load a column from a whitespace-separated signal file. Faster than np.loadtxt."""
    df = pd.read_csv(
        filepath, sep=r"\s+", header=None, usecols=[col], dtype=np.float32, engine="c"
    )
    return df.iloc[:, 0].values


def get_length(filename: str, data_dir: str) -> int:
    """Count lines (no parsing). Much faster than loading full file on network storage."""
    filepath = os.path.join(data_dir, filename)
    with open(filepath, "rb") as f:
        return sum(1 for _ in f)


def get_scaled_t_disrupt(shot_no: int, data_dir: str, t_disrupt: float, max_length: int) -> float:
    if max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {max_length}")
    time = _read_signal_file(os.path.join(data_dir, f"{shot_no}.txt"), col=0)
    return int(np.abs(time - t_disrupt).argmin()) / max_length


def get_means(filename: str, data_dir: str) -> List[float]:
    data = _read_signal_file(os.path.join(data_dir, filename), col=1)
    return [float(np.mean(data)), float(np.mean(data**2))]


def _load_and_pad_base(filename: str, data_dir: str, max_length: int, data: NDArray) -> Tuple[int, NDArray[np.float32]]:
    """Base function for loading and padding."""
    shot_no = int(filename[:-4])
    padded = np.zeros(max_length, dtype=np.float32)
    padded[:min(len(data), max_length)] = data[:min(len(data), max_length)]
    return shot_no, padded


def load_and_pad(filename: str, data_dir: str, max_length: int) -> Tuple[int, NDArray[np.float32]]:
    data = _read_signal_file(os.path.join(data_dir, filename), col=1)
    return _load_and_pad_base(filename, data_dir, max_length, data)


def load_and_pad_norm(
    filename: str, data_dir: str, max_length: int, mean: Optional[float] = None, std: Optional[float] = None
) -> Tuple[int, NDArray[np.float32]]:
    data = _read_signal_file(os.path.join(data_dir, filename), col=1)
    if mean is None or std is None:
        mean, std = float(np.mean(data)), float(np.std(data))
    data = (data - mean) / std if std > 0 else np.zeros_like(data)
    return _load_and_pad_base(filename, data_dir, max_length, data)


def load_and_pad_scale(filename: str, data_dir: str, max_length: int) -> Tuple[int, NDArray[np.float32]]:
    data = _read_signal_file(os.path.join(data_dir, filename), col=1)
    data_min, data_max = np.min(data), np.max(data)
    data = (data - data_min) / (data_max - data_min) if data_max > data_min else np.zeros_like(data)
    return _load_and_pad_base(filename, data_dir, max_length, data)
