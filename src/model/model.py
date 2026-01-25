"""Neural network models and loss functions for disruption prediction."""

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import Dataset
    import torch.nn as nn

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("WARNING: pytorch not installed!")
    pass

from constants import CLASSIFICATION_LOSS, TIME_PREDICTION_LOSS


def loss(outputs: Tensor, labels: Tensor) -> Tensor:
    """
    Combined loss for classification and time prediction.

    Args:
        outputs: (batch_size, 2) - [classification_logits, time_prediction]
        labels: (batch_size, 2) - [binary_class, normalized_time]

    Returns:
        Combined loss (BCE + MSE). MSE only applied to disruptive shots.
    """
    class_loss: Tensor = CLASSIFICATION_LOSS(outputs[:, 0], labels[:, 0])

    disruptive_mask: Tensor = labels[:, 0] == 1
    if disruptive_mask.any():
        time_loss: Tensor = TIME_PREDICTION_LOSS(
            outputs[disruptive_mask, 1], labels[disruptive_mask, 1]
        )
    else:
        time_loss: Tensor = torch.tensor(0.0, device=outputs.device)

    return class_loss + time_loss


class IpDataset(Dataset):
    """PyTorch Dataset for plasma current time series data."""


    def __init__(
        self, data_file: str, labels_file: str, classification: bool = False
    ) -> None:
        """
        Args:
            data_file: Path to preprocessed data tensor (.pt).
            labels_file: Path to labels tensor (.pt).
            classification: If True, return only binary label. If False, return [class, time].
        """
        self.data: Tensor = torch.load(data_file)
        self.labels: Tensor = torch.load(labels_file)
        self.classification: bool = classification

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get sample and label at index."""
        if not self.classification:
            sample: Tensor = self.data[idx]
            label: Tensor = self.labels[idx]
        else:
            sample: Tensor = self.data[idx]
            label: Tensor = self.labels[idx, 0]
        return sample, label


class IpCNN(nn.Module):
    """1D CNN for plasma disruption prediction (classification + time prediction)."""

    def __init__(
        self,
        max_length: int,
        conv1: Tuple[int, int, int] = (16, 9, 4),
        conv2: Tuple[int, int, int] = (32, 5, 2),
        conv3: Tuple[int, int, int] = (64, 3, 1),
        pool_size: int = 4,
        classification: bool = False,
    ) -> None:
        """
        Args:
            max_length: Max input sequence length (for FC layer sizing).
            conv1/conv2/conv3: (out_channels, kernel_size, padding) tuples.
            pool_size: Max pooling kernel size.
            classification: If True, output 1 neuron. If False, output 2 neurons.
        """
        super(IpCNN, self).__init__()
        self.conv1: nn.Conv1d = nn.Conv1d(
            1, conv1[0], kernel_size=conv1[1], stride=1, padding=conv1[2]
        )
        self.conv2: nn.Conv1d = nn.Conv1d(
            conv1[0], conv2[0], kernel_size=conv2[1], stride=1, padding=conv2[2]
        )
        self.conv3: nn.Conv1d = nn.Conv1d(
            conv2[0], conv3[0], kernel_size=conv3[1], stride=1, padding=conv3[2]
        )
        self.pool: nn.MaxPool1d = nn.MaxPool1d(
            kernel_size=pool_size, stride=pool_size, padding=0
        )

        # Compute FC input size dynamically
        with torch.no_grad():
            dummy_input: Tensor = torch.zeros(1, 1, max_length)
            dummy_output: Tensor = self.forward_conv(dummy_input)
            num_features_before_fc: int = dummy_output.numel()

        self.fc1: nn.Linear = nn.Linear(num_features_before_fc, 120)
        self.fc2: nn.Linear = nn.Linear(120, 60)
        self.fc3: nn.Linear = nn.Linear(60, 1 if classification else 2)

    def forward_conv(self, x: Tensor) -> Tensor:
        """Forward through conv+pool layers. Returns flattened features."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass. Returns (batch, 1) or (batch, 2) depending on classification mode."""
        x = x.unsqueeze(1)
        x = self.forward_conv(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
