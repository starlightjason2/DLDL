import numpy as np
from typing import Tuple

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
    import torch.nn as nn
    import torch.nn.functional as F
except:
    print("WARNING: pytorch not installed!")
    pass

classification_loss: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
time_prediction_loss: nn.MSELoss = nn.MSELoss()


def loss(outputs: Tensor, labels: Tensor) -> Tensor:
    # outputs = [classification_logits, time_predictions]
    # labels = [binary_class_labels, normalized_time_steps]
    class_loss: Tensor = classification_loss(outputs[:, 0], labels[:, 0])

    # Apply time prediction loss only to disruptive shots
    disruptive_mask: Tensor = labels[:, 0] == 1
    if disruptive_mask.any():
        time_loss: Tensor = time_prediction_loss(
            outputs[disruptive_mask, 1], labels[disruptive_mask, 1]
        )
    else:
        time_loss: Tensor = torch.tensor(0.0, device=outputs.device)

    return class_loss + time_loss


class IpDataset(Dataset):
    def __init__(self, data_file: str, labels_file: str, classification: bool = False) -> None:
        self.data: Tensor = torch.load(data_file)
        self.labels: Tensor = torch.load(labels_file)
        self.classification: bool = classification

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        if not self.classification:
            sample: Tensor = self.data[idx]
            label: Tensor = self.labels[idx]
        else:
            sample: Tensor = self.data[idx]
            label: Tensor = self.labels[idx, 0]
        return sample, label


class IpCNN(nn.Module):
    def __init__(
        self,
        max_length: int,
        conv1: Tuple[int, int, int] = (16, 9, 4),
        conv2: Tuple[int, int, int] = (32, 5, 2),
        conv3: Tuple[int, int, int] = (64, 3, 1),
        pool_size: int = 4,
        classification: bool = False,
    ) -> None:
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
        self.pool: nn.MaxPool1d = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0)
        # Dynamically determine the correct input size to the first FC layer
        with torch.no_grad():
            dummy_input: Tensor = torch.zeros(
                1, 1, max_length
            )  # Batch size of 1, 1 channel, max_length
            dummy_output: Tensor = self.forward_conv(dummy_input)
            num_features_before_fc: int = (
                dummy_output.numel()
            )  # Total number of features from conv layers

        self.fc1: nn.Linear = nn.Linear(num_features_before_fc, 120)
        self.fc2: nn.Linear = nn.Linear(120, 60)
        if classification:
            self.fc3: nn.Linear = nn.Linear(60, 1)
        else:
            self.fc3: nn.Linear = nn.Linear(
                60, 2
            )  # Two outputs: classification and time of disruption

    def forward_conv(self, x: Tensor) -> Tensor:
        # Forward pass through conv and pool layers, used for initializing fc1
        # print("Initial size:", x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print("After conv1 and pool:", x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print("After conv2 and pool:", x.size())
        x = self.pool(F.relu(self.conv3(x)))
        # print("After conv3 and pool:", x.size())
        x = x.view(x.size(0), -1)
        # print("After flattening:", x.size())
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.forward_conv(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
