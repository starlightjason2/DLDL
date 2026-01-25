"""
Neural network models and loss functions for disruption prediction.

This module contains the core model architectures and loss functions used for
predicting plasma disruptions from current time series data.
"""

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
    Compute combined loss for classification and time prediction tasks.

    This loss function combines binary classification loss (BCE) with time
    prediction loss (MSE) for disruptive shots. The time prediction loss is
    only applied to shots that have disruptions.

    Args:
        outputs: Model predictions tensor of shape (batch_size, 2).
            - outputs[:, 0]: Classification logits
            - outputs[:, 1]: Predicted disruption time (normalized)
        labels: Ground truth labels tensor of shape (batch_size, 2).
            - labels[:, 0]: Binary classification label (0=no disruption, 1=disruption)
            - labels[:, 1]: Normalized disruption time step

    Returns:
        Combined loss tensor (classification_loss + time_prediction_loss).

    Note:
        Time prediction loss is set to 0 if there are no disruptive shots
        in the batch to avoid division by zero or unnecessary computation.
    """
    # outputs = [classification_logits, time_predictions]
    # labels = [binary_class_labels, normalized_time_steps]
    class_loss: Tensor = CLASSIFICATION_LOSS(outputs[:, 0], labels[:, 0])

    # Apply time prediction loss only to disruptive shots
    disruptive_mask: Tensor = labels[:, 0] == 1
    if disruptive_mask.any():
        time_loss: Tensor = TIME_PREDICTION_LOSS(
            outputs[disruptive_mask, 1], labels[disruptive_mask, 1]
        )
    else:
        time_loss: Tensor = torch.tensor(0.0, device=outputs.device)

    return class_loss + time_loss


class IpDataset(Dataset):
    """
    PyTorch Dataset for plasma current time series data.

    This dataset loads preprocessed plasma current signals and their corresponding
    labels. It supports both binary classification mode (disruption/no disruption)
    and regression mode (disruption + time of disruption).

    Attributes:
        data: Tensor containing preprocessed time series data.
        labels: Tensor containing labels (binary classification or [class, time]).
        classification: If True, return only binary classification label.
            If False, return both classification and time prediction labels.
    """

    def __init__(
        self, data_file: str, labels_file: str, classification: bool = False
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_file: Path to the preprocessed data tensor file (.pt).
            labels_file: Path to the labels tensor file (.pt).
            classification: If True, dataset returns only binary classification
                labels. If False, returns both classification and time labels.
        """
        self.data: Tensor = torch.load(data_file)
        self.labels: Tensor = torch.load(labels_file)
        self.classification: bool = classification

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get a single sample and its label.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (sample, label):
                - sample: Time series data tensor.
                - label: If classification=True, binary label tensor.
                    If classification=False, tensor with [class_label, time_label].
        """
        if not self.classification:
            sample: Tensor = self.data[idx]
            label: Tensor = self.labels[idx]
        else:
            sample: Tensor = self.data[idx]
            label: Tensor = self.labels[idx, 0]
        return sample, label


class IpCNN(nn.Module):
    """
    1D Convolutional Neural Network for plasma disruption prediction.

    This CNN processes 1D time series of plasma current signals to predict
    both whether a disruption will occur (classification) and when it will
    occur (time prediction).

    Architecture:
        - 3 convolutional layers with max pooling
        - 2 fully connected layers
        - Output layer: 1 neuron for classification-only, 2 neurons for
          classification + time prediction

    Attributes:
        conv1, conv2, conv3: 1D convolutional layers.
        pool: Max pooling layer.
        fc1, fc2: Fully connected hidden layers.
        fc3: Output layer (size depends on classification mode).
    """

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
        Initialize the CNN model.

        Args:
            max_length: Maximum length of input time series (used to compute
                FC layer input size dynamically).
            conv1: Tuple of (out_channels, kernel_size, padding) for first conv layer.
            conv2: Tuple of (out_channels, kernel_size, padding) for second conv layer.
            conv3: Tuple of (out_channels, kernel_size, padding) for third conv layer.
            pool_size: Kernel size for max pooling layers.
            classification: If True, output only classification (1 neuron).
                If False, output both classification and time (2 neurons).

        Note:
            The FC layer input size is computed dynamically by passing a dummy
            input through the convolutional layers to handle variable input lengths.
        """
        super(IpCNN, self).__init__()
        # Convolutional layers: progressively increase channels, decrease kernel size
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

        # Dynamically determine the correct input size to the first FC layer
        # This is necessary because the output size depends on the input length
        with torch.no_grad():
            dummy_input: Tensor = torch.zeros(1, 1, max_length)
            dummy_output: Tensor = self.forward_conv(dummy_input)
            num_features_before_fc: int = dummy_output.numel()

        # Fully connected layers
        self.fc1: nn.Linear = nn.Linear(num_features_before_fc, 120)
        self.fc2: nn.Linear = nn.Linear(120, 60)
        if classification:
            # Classification-only mode: single output (disruption probability)
            self.fc3: nn.Linear = nn.Linear(60, 1)
        else:
            # Full mode: two outputs (classification + time prediction)
            self.fc3: nn.Linear = nn.Linear(60, 2)

    def forward_conv(self, x: Tensor) -> Tensor:
        """
        Forward pass through convolutional and pooling layers.

        This method is used both during initialization (to compute FC input size)
        and during the main forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, sequence_length).

        Returns:
            Flattened feature tensor ready for FC layers.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten for FC layers
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the entire network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length).

        Returns:
            Output tensor:
                - If classification=True: shape (batch_size, 1) - logits
                - If classification=False: shape (batch_size, 2) - [class_logits, time_prediction]
        """
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, seq_len)
        x = self.forward_conv(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
