"""Neural network models and loss functions for disruption prediction."""

import os
from collections.abc import Sized
from typing import Tuple, cast

from loguru import logger

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from constants import (
    CLASSIFICATION_LOSS,
    TIME_PREDICTION_LOSS,
)
from util.distributed import cleanup, setup
from util.processing import split


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
        self.max_length: int = max_length
        self.classification: bool = classification
        self.logger = logger.bind(name=__name__)

    def __len__(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

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

    def validate_preprocessed_files(
        self, data_path: str, labels_path: str, normalization_type: str
    ) -> None:
        """Validate that preprocessed files exist before training.

        Args:
            data_path: Path to preprocessed dataset file.
            labels_path: Path to preprocessed labels file.
            normalization_type: Normalization type used for error messages.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            missing_files = []
            if not os.path.exists(data_path):
                missing_files.append(f"Dataset: {data_path}")
            if not os.path.exists(labels_path):
                missing_files.append(f"Labels: {labels_path}")

            self.logger.error(
                f"Preprocessed files not found for NORMALIZATION_TYPE='{normalization_type}'. "
                f"Please run preprocess_data.py first with the same NORMALIZATION_TYPE.\n"
                f"Missing files:\n"
                + "\n".join(f"  - {f}" for f in missing_files)
                + "\n"
                f"Note: The NORMALIZATION_TYPE in both preprocess_data.py and train.py must match."
            )
            raise FileNotFoundError(
                f"Preprocessed files not found for NORMALIZATION_TYPE='{normalization_type}'. "
                f"Run preprocess_data.py first with matching NORMALIZATION_TYPE."
            )

    def _validate_epoch(
        self,
        model: nn.Module,
        dev_loader: DataLoader,
        classification: bool,
        bce_loss: nn.Module,
        mse_loss: nn.Module | None,
        loss_fn,
        epoch: int,
        writer: SummaryWriter,
        total_train_loss: float,
        train_loader: DataLoader,
        logs: list,
    ) -> None:
        """Run validation for a single epoch and update logs.

        Args:
            model: The model to validate (may be DDP wrapped).
            dev_loader: Validation data loader.
            classification: Whether in classification-only mode.
            bce_loss: Binary cross-entropy loss function.
            mse_loss: Mean squared error loss function (None if classification=True).
            loss_fn: Combined loss function.
            epoch: Current epoch number.
            writer: TensorBoard writer.
            total_train_loss: Total training loss for the epoch.
            train_loader: Training data loader (for computing average).
            logs: List of log dictionaries to append to.
        """
        model.eval()
        total_val_loss = 0.0
        if not classification:
            all_classification_targets, all_classification_predictions = [], []
            all_time_targets, all_time_predictions = [], []

        with torch.no_grad():
            for data, targets in dev_loader:
                data, targets = data.cuda(), targets.cuda()
                output = model(data)
                if not classification:
                    classification_targets, time_targets = (
                        targets[:, 0],
                        targets[:, 1],
                    )
                    classification_output, time_output = (
                        output[:, 0],
                        output[:, 1],
                    )
                    val_loss_classification = bce_loss(
                        classification_output, classification_targets
                    )
                    val_loss_time = mse_loss(time_output, time_targets)
                    all_time_targets.extend(time_targets.cpu().numpy())
                    all_time_predictions.extend(time_output.cpu().numpy())
                else:
                    classification_output = output
                    classification_targets = targets

                classification_predictions = (
                    torch.sigmoid(classification_output) > 0.5
                )
                all_classification_targets.extend(
                    classification_targets.cpu().numpy()
                )
                all_classification_predictions.extend(
                    classification_predictions.cpu().numpy()
                )

                val_total_loss = loss_fn(output, targets)
                total_val_loss += val_total_loss.item()

        # Compute average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(dev_loader)
        logs.append(
            {
                "epoch": epoch,
                "training_loss": avg_train_loss,
                "validation_loss": avg_val_loss,
                "Validation Accuracy": accuracy_score(
                    all_classification_targets, all_classification_predictions
                ),
                "Validation Precision": precision_score(
                    all_classification_targets, all_classification_predictions
                ),
                "Validation Recall": recall_score(
                    all_classification_targets, all_classification_predictions
                ),
                "Validation F1 Score": f1_score(
                    all_classification_targets, all_classification_predictions
                ),
            }
        )

        writer.add_scalar("Validation Loss", avg_val_loss, epoch)
        writer.add_scalar(
            "Validation Accuracy",
            accuracy_score(
                all_classification_targets, all_classification_predictions
            ),
            epoch,
        )
        writer.add_scalar(
            "Validation Precision",
            precision_score(
                all_classification_targets, all_classification_predictions
            ),
            epoch,
        )
        writer.add_scalar(
            "Validation Recall",
            recall_score(
                all_classification_targets, all_classification_predictions
            ),
            epoch,
        )
        writer.add_scalar(
            "Validation F1 Score",
            f1_score(
                all_classification_targets, all_classification_predictions
            ),
            epoch,
        )
        if not classification:
            writer.add_scalar(
                "Validation Time MSE",
                mse_loss(
                    torch.tensor(all_time_predictions),
                    torch.tensor(all_time_targets),
                ).item(),
                epoch,
            )

    def train_model(
        self,
        rank: int,
        world_size: int,
        data_path: str,
        labels_path: str,
        prog_dir: str,
        job_id: str,
        normalization_type: str = "",
        lr: float = 0.01,
        num_epochs: int = 100,
        log_interval: int = 20,
    ) -> None:
        """
        Train this model with distributed data parallel.

        Args:
            rank: Process rank (0 to world_size-1).
            world_size: Total number of processes/GPUs.
            data_path: Path to preprocessed dataset (.pt).
            labels_path: Path to labels (.pt).
            prog_dir: Directory for logs and checkpoints.
            job_id: Training run identifier.
            normalization_type: Normalization type used for validation messages.
            lr: Learning rate (default: 0.01).
            num_epochs: Number of epochs (default: 100).
            log_interval: Batches between logging (default: 20).

        Note: Only rank 0 performs validation, logging, and checkpointing.
        """
        self.validate_preprocessed_files(data_path, labels_path, normalization_type)

        self.logger.info(f"Loading preprocessed dataset from {data_path}")
        self.logger.info(f"Using labels from {labels_path}")
        if normalization_type:
            self.logger.info(f"Normalization type: {normalization_type}")

        self.logger.info(f"GPUs Available: {torch.cuda.device_count()}")
        use_distributed = world_size > 1
        if use_distributed:
            self.logger.info(
                f"Distributed training - Rank: {rank}, World Size: {world_size}"
            )
            setup(rank, world_size)
        else:
            self.logger.info("Single-process training (world_size=1)")
        torch.manual_seed(42 + rank)

        dataset_obj = IpDataset(data_path, labels_path, self.classification)
        self.logger.info(
            f"Dataset loaded: {len(dataset_obj)} examples, max_length={self.max_length}"
        )
        train, dev, _ = split(dataset_obj)

        if use_distributed:
            train_sampler = DistributedSampler(
                train, num_replicas=world_size, rank=rank, shuffle=True
            )
            train_loader = DataLoader(
                train, batch_size=128, sampler=train_sampler, pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train, batch_size=128, shuffle=True, pin_memory=True
            )
        dev_loader = DataLoader(dev, batch_size=128, shuffle=False, pin_memory=True)

        model = self.cuda()
        if use_distributed:
            model = DDP(model, device_ids=[0])

        optimizer = optim.Adam(model.parameters(), lr=lr)
        bce_loss = torch.nn.BCEWithLogitsLoss()
        classification = self.classification
        if not classification:
            mse_loss = torch.nn.MSELoss()

        logs = []
        if rank == 0:
            writer = SummaryWriter(os.path.join(prog_dir, job_id))

        for epoch in range(num_epochs):
            if rank == 0 and epoch > 0:
                self.logger.info("--------------------------------")

            model.train()
            total_train_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()

                if not classification:
                    classification_targets, time_targets = target[:, 0], target[:, 1]

                optimizer.zero_grad()
                output = model(data)

                if not classification:
                    classification_output, time_output = output[:, 0], output[:, 1]
                    loss_classification = bce_loss(
                        classification_output, classification_targets
                    )
                    loss_time = mse_loss(time_output, time_targets)
                    loss_value = loss(output, target)
                else:
                    loss_value = bce_loss(output, target)

                loss_value.backward()
                optimizer.step()
                total_train_loss += loss_value.item()

                if batch_idx % log_interval == 0:
                    dataset_size = len(cast(Sized, train_loader.dataset))
                    if rank == 0:
                        self.logger.info(
                            f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}, "
                            f"[{batch_idx * len(data)}/{dataset_size}] "
                            f"Loss {loss_value.item():.6f}"
                        )
                    if rank == 0:
                        writer.add_scalar(
                            "Training Loss",
                            loss_value.item(),
                            epoch * dataset_size + batch_idx,
                        )
                        if not classification:
                            writer.add_scalar(
                                "Training Classification Loss",
                                loss_classification.item(),
                                epoch * dataset_size + batch_idx,
                            )
                            writer.add_scalar(
                                "Training Time Loss",
                                loss_time.item(),
                                epoch * dataset_size + batch_idx,
                            )

            if rank == 0:
                self._validate_epoch(
                    model=model,
                    dev_loader=dev_loader,
                    classification=classification,
                    bce_loss=bce_loss,
                    mse_loss=mse_loss if not classification else None,
                    loss_fn=loss,
                    epoch=epoch,
                    writer=writer,
                    total_train_loss=total_train_loss,
                    train_loader=train_loader,
                    logs=logs,
                )

            if epoch % 5 == 0 and rank == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(prog_dir, f"{job_id}_params_epoch{epoch}.pt"),
                )

        if rank == 0:
            writer.close()
            df_logs = pd.DataFrame(logs)
            df_logs.to_csv(
                os.path.join(prog_dir, f"{job_id}_training_log.csv"), index=False
            )

        if use_distributed:
            cleanup()

