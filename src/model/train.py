"""Distributed training for DLDL disruption prediction model."""

import os
from typing import cast
from collections.abc import Sized
from constants import LOCAL_RANK

if LOCAL_RANK:
    os.environ["CUDA_VISIBLE_DEVICES"] = LOCAL_RANK
from .model import IpDataset, IpCNN, loss
from util.dataset import split
from util.distributed import setup, setup_file, cleanup
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
    Train IpCNN model with distributed data parallel.

    Args:
        rank: Process rank (0 to world_size-1).
        world_size: Total number of processes/GPUs.
        data_path: Path to preprocessed dataset (.pt).
        labels_path: Path to labels (.pt).
        prog_dir: Directory for logs and checkpoints.
        max_length: Max sequence length.
        job_id: Training run identifier.
        lr: Learning rate (default: 0.01).
        num_epochs: Number of epochs (default: 100).
        log_interval: Batches between logging (default: 20).
        classification: If True, classification-only. If False, classification + time.

    Note: Only rank 0 performs validation, logging, and checkpointing.
    """
    setup(rank, world_size)
    torch.manual_seed(42 + rank)

    dataset = IpDataset(data_path, labels_path, classification)
    train, dev, _ = split(dataset)

    train_sampler = DistributedSampler(
        train, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train, batch_size=128, sampler=train_sampler, pin_memory=True
    )
    dev_loader = DataLoader(dev, batch_size=128, shuffle=False, pin_memory=True)

    model = IpCNN(max_length=max_length, classification=classification).cuda()
    model = DDP(model, device_ids=[0])

    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    if not classification:
        mse_loss = torch.nn.MSELoss()

    logs = []
    if rank == 0:
        writer = SummaryWriter(prog_dir + job_id)

    for epoch in range(num_epochs):
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

        if rank == 0:
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

        if epoch % 5 == 0 and rank == 0:
            torch.save(model.state_dict(), f"{prog_dir}{job_id}_params_epoch{epoch}.pt")

    if rank == 0:
        writer.close()
        df_logs = pd.DataFrame(logs)
        df_logs.to_csv(prog_dir + job_id + "_training_log.csv", index=False)

    cleanup()
