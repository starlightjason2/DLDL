"""
Utility functions for DLDL project.
"""

import os
from typing import Tuple, List, Optional, Any, TYPE_CHECKING, cast
from collections.abc import Sized
import numpy as np
from numpy.typing import NDArray
from datetime import timedelta

try:
    import torch
    from torch.utils.data import Dataset, Subset
    import torch.distributed as dist
except:
    pass


################################################################################
## File I/O Utilities
################################################################################
def check_file(file_path: str, verbose: bool = False) -> bool:
    """
    Check if a file exists.

    Args:
        file_path: str, path to file
        verbose: bool, if True, print file size or non-existence message

    Returns:
        bool: True if file exists, False otherwise
    """
    if os.path.exists(file_path):
        if verbose:
            file_size: int = os.path.getsize(file_path)
            print(f"File {file_path} exists. Size: {file_size} bytes.")
        return True
    else:
        if verbose:
            print(f"File {file_path} does not exist.")
        return False


################################################################################
## Data Loading Utilities
################################################################################
def get_length(filename: str, data_dir: str) -> int:
    """
    Get length of time series for a single file.

    Args:
        filename: str, file name
        data_dir: str, path to file directory

    Returns:
        int: length of the time series
    """
    file_path: str = os.path.join(data_dir, filename)
    data: NDArray[np.float64] = np.loadtxt(file_path, usecols=1)

    return len(data)


def get_scaled_t_disrupt(
    shot_no: int, data_dir: str, t_disrupt: float, max_length: int
) -> float:
    """
    Compute normalized disruption time index for a shot.

    Finds the time index closest to the disruption time and normalizes it
    by the maximum sequence length to produce a value in [0, 1].

    Args:
        shot_no: Shot number to process.
        data_dir: Directory containing shot data files.
        t_disrupt: Actual disruption time (in seconds or time units).
        max_length: Maximum sequence length for normalization.

    Returns:
        Normalized disruption index in [0, 1] range.
            Represents the fraction of the sequence where disruption occurs.
    """
    shot_file: str = os.path.join(data_dir, str(shot_no) + ".txt")
    time: NDArray[np.float64] = np.loadtxt(shot_file, usecols=0)

    # Find the time index closest to the disruption time
    disruption_index: int = int(np.abs(time - t_disrupt).argmin())

    # Return normalized index in [0, 1] range
    return disruption_index / max_length


def get_means(filename: str, data_dir: str) -> List[float]:
    """
    Compute mean and mean of squares for a time series file.

    These statistics are used to compute dataset-wide mean and standard
    deviation efficiently. The std can be computed as:
    std = sqrt(E[X^2] - E[X]^2)

    Args:
        filename: Name of the signal file (e.g., "12345.txt").
        data_dir: Directory containing the signal files.

    Returns:
        List containing [mean, mean_squared] where:
            - mean: Average value of the time series
            - mean_squared: Average of squared values (E[X^2])
    """
    file_path: str = os.path.join(data_dir, filename)
    data: NDArray[np.float64] = np.loadtxt(file_path, usecols=1)

    mean: float = np.mean(data)
    mean_squared: float = np.mean(data**2)

    return [mean, mean_squared]


def load_and_pad(
    filename: str, data_dir: str, max_length: int
) -> Tuple[int, NDArray[np.float32]]:
    """
    Load a time series file and pad with zeros to a fixed length.

    This function loads raw signal data and ensures all sequences have
    the same length by padding shorter sequences with zeros.

    Args:
        filename: Name of the signal file (e.g., "12345.txt").
        data_dir: Directory containing the signal files.
        max_length: Target length for padding. Sequences longer than this
            are truncated.

    Returns:
        Tuple of (shot_number, padded_data):
            - shot_number: Integer shot number extracted from filename
            - padded_data: NumPy array of shape (max_length,) with zero padding
    """
    shot_no: int = int(filename[:-4])
    file_path: str = os.path.join(data_dir, filename)
    data: NDArray[np.float32] = np.loadtxt(file_path, usecols=1, dtype=np.float32)
    length: int = min(len(data), max_length)
    padded_data: NDArray[np.float32] = np.zeros(max_length, dtype=np.float32)
    padded_data[:length] = data

    return (shot_no, padded_data)


def load_and_pad_norm(
    filename: str,
    data_dir: str,
    max_length: int,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[int, NDArray[np.float32]]:
    """
    Load, normalize (Z-score), and pad a time series file.

    Applies Z-score normalization: (x - mean) / std. If mean/std are not
    provided, computes per-shot statistics. Then pads to max_length.

    Args:
        filename: Name of the signal file (e.g., "12345.txt").
        data_dir: Directory containing the signal files.
        max_length: Target length for padding.
        mean: Dataset-wide mean for normalization. If None, uses per-shot mean.
        std: Dataset-wide std for normalization. If None, uses per-shot std.

    Returns:
        Tuple of (shot_number, normalized_padded_data):
            - shot_number: Integer shot number
            - normalized_padded_data: Z-score normalized and zero-padded array
    """
    shot_no: int = int(filename[:-4])
    file_path: str = os.path.join(data_dir, filename)
    data: NDArray[np.float32] = np.loadtxt(file_path, usecols=1, dtype=np.float32)

    if mean is None or std is None:
        mean = float(np.mean(data))
        std = float(np.std(data))

    data = (data - mean) / std

    length: int = min(len(data), max_length)
    padded_data: NDArray[np.float32] = np.zeros(max_length, dtype=np.float32)
    padded_data[:length] = data

    return (shot_no, padded_data)


def load_and_pad_scale(
    filename: str, data_dir: str, max_length: int
) -> Tuple[int, NDArray[np.float32]]:
    """
    Load, min-max scale to [0,1], and pad a time series file.

    Applies min-max normalization per shot: (x - min) / (max - min).
    This ensures all values are in [0, 1] range. Then pads to max_length.

    Args:
        filename: Name of the signal file (e.g., "12345.txt").
        data_dir: Directory containing the signal files.
        max_length: Target length for padding.

    Returns:
        Tuple of (shot_number, scaled_padded_data):
            - shot_number: Integer shot number
            - scaled_padded_data: Min-max scaled and zero-padded array
    """
    shot_no: int = int(filename[:-4])
    file_path: str = os.path.join(data_dir, filename)
    data: NDArray[np.float32] = np.loadtxt(file_path, usecols=1, dtype=np.float32)

    # Min-max scaling to [0, 1] range
    data = data - np.min(data)
    data = data / np.max(data)

    # Pad to fixed length
    length: int = min(len(data), max_length)
    padded_data: NDArray[np.float32] = np.zeros(max_length, dtype=np.float32)
    padded_data[:length] = data

    return (shot_no, padded_data)


################################################################################
## Dataset Utilities
################################################################################
def split(
    dataset: "Dataset", train_size: float = 0.8
) -> Tuple["Subset", "Subset", "Subset"]:
    """
    Split a dataset into train, dev, and test sets.

    Args:
        dataset: Dataset object to split (must implement __len__)
        train_size: float, fraction of data for training (default 0.8)

    Returns:
        tuple: (train, dev, test) Subset objects
    """
    dev_size: float = (1 - train_size) / 2

    # Cast to Sized for len() call (Dataset implements __len__ but type stubs don't reflect this)
    total_size: int = len(cast(Sized, dataset))
    train_end: int = int(train_size * total_size)
    dev_end: int = int((train_size + dev_size) * total_size)
    train_indices = range(0, train_end)
    dev_indices = range(train_end, dev_end)
    test_indices = range(dev_end, total_size)

    train: "Subset" = Subset(dataset, train_indices)
    dev: "Subset" = Subset(dataset, dev_indices)
    test: "Subset" = Subset(dataset, test_indices)

    return train, dev, test


################################################################################
## Distributed Training Utilities
################################################################################
def setup(rank: int, world_size: int) -> None:
    """
    Initialize distributed training process group.

    Args:
        rank: int, rank of the current process
        world_size: int, total number of processes
    """
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=10),
    )
    torch.cuda.set_device(0)  # Assign a GPU to each process
    # Each process sees only one GPU, so use ID 0


def setup_file(rank: int, world_size: int, rendezvous_file: str) -> None:
    """
    Initialize distributed training process group using file-based rendezvous.

    Args:
        rank: int, rank of the current process
        world_size: int, total number of processes
        rendezvous_file: str, path to rendezvous file
    """
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=10),
    )
    torch.cuda.set_device(rank)  # Assign a GPU to each process


def cleanup() -> None:
    """Destroy the distributed process group."""
    dist.destroy_process_group()
