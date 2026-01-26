"""CNN model for disruption prediction."""

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
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from constants import (
    CLASSIFICATION_LOSS,
    TIME_PREDICTION_LOSS,
)
from util.distributed import cleanup, setup
from util.processing import split
from model.dataset import IpDataset


class IpCNN(nn.Module):
    """1D CNN for plasma disruption prediction (classification + time prediction)."""

    def __init__(
        self,
        data_path: str,
        labels_path: str,
        prog_dir: str,
        conv1: Tuple[int, int, int] = (16, 9, 4),
        conv2: Tuple[int, int, int] = (32, 5, 2),
        conv3: Tuple[int, int, int] = (64, 3, 1),
        pool_size: int = 4,
        classification: bool = False,
    ) -> None:
        """Initialize CNN model."""
        super(IpCNN, self).__init__()

        # Load dataset to get max_length (will be loaded again in train_model for validation)
        dataset = torch.load(data_path)
        self.max_length = int(dataset.shape[1])
        del dataset  # Free memory before training

        self.labels_path = labels_path
        self.data_path = data_path
        self.prog_dir = prog_dir
        self.classification = classification
        self.logger = logger.bind(name=__name__)

        self.conv1 = nn.Conv1d(1, conv1[0], kernel_size=conv1[1], stride=1, padding=conv1[2])
        self.bn1 = nn.BatchNorm1d(conv1[0])
        self.conv2 = nn.Conv1d(conv1[0], conv2[0], kernel_size=conv2[1], stride=1, padding=conv2[2])
        self.bn2 = nn.BatchNorm1d(conv2[0])
        self.conv3 = nn.Conv1d(conv2[0], conv3[0], kernel_size=conv3[1], stride=1, padding=conv3[2])
        self.bn3 = nn.BatchNorm1d(conv3[0])
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0)

        # Compute FC input size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.max_length)
            dummy_output = self.forward_conv(dummy_input)
            num_features_before_fc = dummy_output.numel()

        self.fc1 = nn.Linear(num_features_before_fc, 120)
        self.bn4 = nn.BatchNorm1d(120)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 60)
        self.bn5 = nn.BatchNorm1d(60)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(60, 1 if classification else 2)

    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_conv(self, x: Tensor) -> Tensor:
        """Forward through conv+pool layers."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x.view(x.size(0), -1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.forward_conv(x.unsqueeze(1))
        x = self.dropout1(F.relu(self.bn4(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn5(self.fc2(x))))
        return self.fc3(x)

    def validate_preprocessed_files(self, normalization_type: str) -> None:
        """Validate that preprocessed files exist."""
        if not os.path.exists(self.data_path) or not os.path.exists(self.labels_path):
            missing = [f"Dataset: {self.data_path}" if not os.path.exists(self.data_path) else None,
                      f"Labels: {self.labels_path}" if not os.path.exists(self.labels_path) else None]
            missing = [m for m in missing if m]
            self.logger.error(f"Preprocessed files not found for NORMALIZATION_TYPE='{normalization_type}'. Missing: {', '.join(missing)}")
            raise FileNotFoundError("Preprocessed files not found. Run preprocess_data.py first.")

    def _loss(self, outputs: Tensor, labels: Tensor) -> Tensor:
        """Combined loss for classification and time prediction."""
        class_loss = CLASSIFICATION_LOSS(outputs[:, 0], labels[:, 0])
        disruptive_mask = labels[:, 0] == 1
        time_loss = TIME_PREDICTION_LOSS(outputs[disruptive_mask, 1], labels[disruptive_mask, 1]) if disruptive_mask.any() else torch.tensor(0.0, device=outputs.device)
        return class_loss + time_loss

    def _validate_epoch(self, model: nn.Module, dev_loader: DataLoader, classification: bool, bce_loss: nn.Module, mse_loss: nn.Module | None, epoch: int, writer: SummaryWriter, total_train_loss: float, train_loader: DataLoader, logs: list) -> None:
        """Run validation for a single epoch and update logs."""
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
                    classification_targets, time_targets = targets[:, 0], targets[:, 1]
                    classification_output, time_output = output[:, 0], output[:, 1]
                    all_time_targets.extend(time_targets.cpu().numpy())
                    all_time_predictions.extend(time_output.cpu().numpy())
                else:
                    classification_output, classification_targets = output, targets

                classification_predictions = torch.sigmoid(classification_output) > 0.5
                all_classification_targets.extend(classification_targets.cpu().numpy())
                all_classification_predictions.extend(classification_predictions.cpu().numpy())

                val_total_loss = self._loss(output, targets)
                total_val_loss += val_total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(dev_loader)
        metrics = {
            "Validation Accuracy": accuracy_score(all_classification_targets, all_classification_predictions),
            "Validation Precision": precision_score(all_classification_targets, all_classification_predictions),
            "Validation Recall": recall_score(all_classification_targets, all_classification_predictions),
            "Validation F1 Score": f1_score(all_classification_targets, all_classification_predictions),
        }
        logs.append({"epoch": epoch, "training_loss": avg_train_loss, "validation_loss": avg_val_loss, **metrics})
        for name, value in metrics.items():
            writer.add_scalar(name, value, epoch)
        writer.add_scalar("Validation Loss", avg_val_loss, epoch)
        if not classification:
            writer.add_scalar(
                "Validation Time MSE",
                mse_loss(
                    torch.tensor(all_time_predictions),
                    torch.tensor(all_time_targets),
                ).item(),
                epoch,
            )

    def train_model(self, rank: int, world_size: int, job_id: str, normalization_type: str = "", lr: float = 0.01, num_epochs: int = 100, log_interval: int = 20, weight_decay: float = 1e-4, lr_scheduler: bool = True, early_stopping_patience: int = 10, gradient_clip: float = 1.0) -> None:
        """Train this model with distributed data parallel."""
        self.validate_preprocessed_files(normalization_type)

        self.logger.info(f"Loading preprocessed dataset from {self.data_path}")
        self.logger.info(f"Using labels from {self.labels_path}")
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

        dataset_obj = IpDataset(
            data_file=self.data_path,
            labels_file=self.labels_path,
            classification=self.classification,
        )
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

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        bce_loss = torch.nn.BCEWithLogitsLoss()
        classification = self.classification
        if not classification:
            mse_loss = torch.nn.MSELoss()

        logs = []
        if rank == 0:
            writer = SummaryWriter(os.path.join(self.prog_dir, job_id))

        best_val_loss = float("inf")
        epochs_without_improvement = 0

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
                    loss_value = self._loss(output, target)
                else:
                    loss_value = bce_loss(output, target)

                loss_value.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                total_train_loss += loss_value.item()

                if batch_idx % log_interval == 0 and rank == 0:
                    dataset_size = len(cast(Sized, train_loader.dataset))
                    step = epoch * dataset_size + batch_idx
                    self.logger.info(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}, [{batch_idx * len(data)}/{dataset_size}] Loss {loss_value.item():.6f}")
                    writer.add_scalar("Training Loss", loss_value.item(), step)
                    if not classification:
                        writer.add_scalar("Training Classification Loss", loss_classification.item(), step)
                        writer.add_scalar("Training Time Loss", loss_time.item(), step)

            if rank == 0:
                self._validate_epoch(
                    model=model,
                    dev_loader=dev_loader,
                    classification=classification,
                    bce_loss=bce_loss,
                    mse_loss=mse_loss if not classification else None,
                    epoch=epoch,
                    writer=writer,
                    total_train_loss=total_train_loss,
                    train_loader=train_loader,
                    logs=logs,
                )
                avg_val_loss = logs[-1]["validation_loss"] if logs else float("inf")
                
                if lr_scheduler:
                    scheduler.step(avg_val_loss)
                    current_lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("Learning Rate", current_lr, epoch)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.prog_dir, f"{job_id}_best_params.pt"),
                    )
                    self.logger.info(f"New best validation loss: {best_val_loss:.6f}")
                else:
                    epochs_without_improvement += 1
                
                if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {early_stopping_patience} epochs)")
                    break

            if epoch % 5 == 0 and rank == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(self.prog_dir, f"{job_id}_params_epoch{epoch}.pt"),
                )

        if rank == 0:
            writer.close()
            df_logs = pd.DataFrame(logs)
            df_logs.to_csv(os.path.join(self.prog_dir, f"{job_id}_training_log.csv"), index=False)

        if use_distributed:
            cleanup()
