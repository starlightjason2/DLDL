"""Model module: neural network models, datasets, and training functionality."""

from .cnn import IpCNN
from .dataset import IpDataset

__all__ = [
    "IpDataset",
    "IpCNN",
]
