"""
Distributed training script for DLDL disruption prediction model.

This module provides the main training function that supports distributed
multi-GPU training using PyTorch's DistributedDataParallel.
"""

import os
from typing import cast
from collections.abc import Sized
from constants import LOCAL_RANK

if LOCAL_RANK:
    os.environ["CUDA_VISIBLE_DEVICES"] = LOCAL_RANK
from .model import IpDataset, IpCNN, loss
from util.utils import split, setup, setup_file, cleanup
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def train(
    rank: int,
    world_size: int,
    data_path: str,
    labels_path: str,
    prog_dir: str,
    max_length: int,
    job_id: str,
    lr: float = 0.01,
    num_epochs: int = 100,
    log_interval: int = 20,
    classification: bool = True,
) -> None:
    """
    Train the IpCNN model using distributed data parallel training.

    This function handles the complete training loop including:
    - Distributed process group setup
    - Dataset loading and splitting
    - Model initialization and DDP wrapping
    - Training and validation loops
    - Metrics logging (TensorBoard and CSV)
    - Model checkpointing

    Args:
        rank: Process rank in distributed training (0 to world_size-1).
        world_size: Total number of processes/GPUs.
        data_path: Path to preprocessed dataset tensor file (.pt).
        labels_path: Path to labels tensor file (.pt).
        prog_dir: Directory to save training progress, logs, and checkpoints.
        max_length: Maximum sequence length of input time series.
        job_id: Unique identifier for this training run (used in filenames).
        lr: Learning rate for Adam optimizer. Default: 0.01.
        num_epochs: Number of training epochs. Default: 100.
        log_interval: Number of batches between logging training progress. Default: 20.
        classification: If True, train in classification-only mode.
            If False, train for both classification and time prediction.

    Note:
        - Only rank 0 process performs validation, logging, and checkpointing.
        - Model checkpoints are saved every 5 epochs.
        - Training logs are saved to both TensorBoard and CSV format.
    """
    setup(rank, world_size)

    # Set different random seed for each process to ensure data diversity
    torch.manual_seed(42 + rank)

    # Load and split dataset
    dataset = IpDataset(data_path, labels_path, classification)
    train, dev, _ = split(dataset)

    # Create distributed sampler for training data
    # This ensures each process sees a different subset of the data
    train_sampler = DistributedSampler(
        train, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train, batch_size=128, sampler=train_sampler, pin_memory=True
    )
    dev_loader = DataLoader(dev, batch_size=128, shuffle=False, pin_memory=True)

    # Initialize model and wrap with DistributedDataParallel
    model = IpCNN(max_length=max_length, classification=classification).cuda()
    model = DDP(model, device_ids=[0])

    # Initialize optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce_loss = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy for classification
    if not classification:
        mse_loss = torch.nn.MSELoss()  # Mean squared error for time prediction

    # Initialize logging (only on rank 0 to avoid duplicate logs)
    logs = []
    if rank == 0:
        writer = SummaryWriter(prog_dir + job_id)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to GPU
            data, target = data.cuda(), target.cuda()

            # Extract targets for multi-task learning (if applicable)
            if not classification:
                classification_targets, time_targets = target[:, 0], target[:, 1]

            # Forward pass
            optimizer.zero_grad()
            output = model(data)

            # Compute loss based on task mode
            if not classification:
                # Multi-task: separate classification and time prediction outputs
                classification_output, time_output = output[:, 0], output[:, 1]
                loss_classification = bce_loss(
                    classification_output, classification_targets
                )
                loss_time = mse_loss(time_output, time_targets)
                # Combined loss (defined in model.loss function)
                loss_value = loss(output, target)
            else:
                # Classification-only: single output
                loss_value = bce_loss(output, target)

            # Backward pass and optimization
            loss_value.backward()
            optimizer.step()
            total_train_loss += loss_value.item()

            if batch_idx % log_interval == 0:
                dataset_size = len(cast(Sized, train_loader.dataset))
                print(
                    f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, "
                    + f"[{batch_idx * len(data)}/{dataset_size}] "
                    + f"Loss {loss_value.item()}"
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

        # Validation and logging (only on rank 0 to avoid duplicate work)
        if rank == 0:
            model.eval()
            total_val_loss = 0.0

            # Accumulate predictions for metric computation
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
                        classification_output, time_output = output[:, 0], output[:, 1]
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

                    val_total_loss = loss(output, targets)
                    total_val_loss += val_total_loss.item()

            # Compute average losses
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(dev_loader)

            # Log metrics to DataFrame for CSV export
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
                f1_score(all_classification_targets, all_classification_predictions),
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

        # Save model checkpoint every 5 epochs
        if epoch % 5 == 0 and rank == 0:
            torch.save(model.state_dict(), f"{prog_dir}{job_id}_params_epoch{epoch}.pt")

    # Finalize logging and cleanup
    if rank == 0:
        writer.close()
        # Save training logs to CSV for analysis
        df_logs = pd.DataFrame(logs)
        df_logs.to_csv(prog_dir + job_id + "_training_log.csv", index=False)

    cleanup()
