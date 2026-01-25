"""
Model module for DLDL project.

This package contains the neural network models, datasets, and training
functionality for plasma disruption prediction.

Exports:
    - IpDataset: PyTorch Dataset for loading preprocessed time series
    - IpCNN: 1D CNN model for disruption prediction
    - loss: Combined loss function for classification and time prediction
    - train: Distributed training function
"""

from .model import IpDataset, IpCNN, loss
from .train import train

__all__ = [
    "IpDataset",
    "IpCNN",
    "loss",
    "train",
]
