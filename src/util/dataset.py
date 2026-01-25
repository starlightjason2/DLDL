"""Dataset utilities for DLDL project."""

from typing import Tuple, TYPE_CHECKING, cast
from collections.abc import Sized

try:
    from torch.utils.data import Dataset, Subset
except ImportError:
    pass

if TYPE_CHECKING:
    from torch.utils.data import Dataset, Subset


def split(
    dataset: "Dataset", train_size: float = 0.8
) -> Tuple["Subset", "Subset", "Subset"]:
    """Split dataset into train, dev, and test sets. Returns (train, dev, test)."""
    dev_size: float = (1 - train_size) / 2
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
