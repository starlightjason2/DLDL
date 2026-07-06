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
    fbeta_score,
)
from torch import Tensor
from torch.utils.data import DataLoader

from model.dataset import IpDataset
from util.objective import (
    INFEASIBLE_SCORE,
    PRECISION_COL,
    RECALL_COL,
    THRESHOLD_COL,
    best_row,
    min_precision,
    score,
    validation_metrics,
)


class IpCNN(nn.Module):
    """1D CNN for plasma disruption prediction (binary disruption classification)."""

    def __init__(
        self,
        dataset: IpDataset,
        prog_dir: str,
        conv1: Tuple[int, int, int],
        conv2: Tuple[int, int, int],
        conv3: Tuple[int, int, int],
        conv4: Tuple[int, int, int],
        pool_size: int,
        fc1_size: int,
        fc2_size: int,
        dropout_rate: float,
        cls_pos_weight: float = 1.0,
        decision_threshold: float = 0.5,
    ) -> None:
        """Initialize CNN model.

        ``cls_pos_weight`` scales the loss of the positive (disruptive) class in
        the binary classification head. Values > 1 penalize false negatives
        (missed disruptions) more heavily, trading precision for recall.

        ``decision_threshold`` is the sigmoid probability cutoff for predicting
        the disruptive class. Values < 0.5 classify more shots as disruptive,
        raising recall at the cost of precision.
        """
        super(IpCNN, self).__init__()
        self.logger = logger.bind(name=__name__)
        self.prog_dir = prog_dir
        self.cls_pos_weight = float(cls_pos_weight)
        self.decision_threshold = float(decision_threshold)
        self._cls_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.cls_pos_weight)
        )

        # Log hyperparameters
        self.logger.info("=" * 60)
        self.logger.info("CNN Architecture Hyperparameters:")
        self.logger.info(
            f"  Conv1: filters={conv1[0]}, kernel={conv1[1]}, padding={conv1[2]}"
        )
        self.logger.info(
            f"  Conv2: filters={conv2[0]}, kernel={conv2[1]}, padding={conv2[2]}"
        )
        self.logger.info(
            f"  Conv3: filters={conv3[0]}, kernel={conv3[1]}, padding={conv3[2]}"
        )
        self.logger.info(
            f"  Conv4: filters={conv4[0]}, kernel={conv4[1]}, padding={conv4[2]}"
        )
        self.logger.info(f"  Pool size: {pool_size}")
        self.logger.info(f"  FC1 size: {fc1_size}")
        self.logger.info(f"  FC2 size: {fc2_size}")
        self.logger.info(f"  Dropout rate: {dropout_rate}")
        self.logger.info(f"  Classification pos_weight: {self.cls_pos_weight}")
        self.logger.info(f"  Decision threshold: {self.decision_threshold}")
        self.logger.info("=" * 60)

        self.dataset = dataset
        self.max_length = int(self.dataset.data.shape[1])
        self.logger.info(
            f"Dataset: {len(self.dataset)} examples, max_length={self.max_length}"
        )

        # create CNN layers
        self.conv1 = nn.Conv1d(
            1, conv1[0], kernel_size=conv1[1], stride=1, padding=conv1[2]
        )
        self.bn1 = nn.BatchNorm1d(conv1[0])
        self.conv2 = nn.Conv1d(
            conv1[0], conv2[0], kernel_size=conv2[1], stride=1, padding=conv2[2]
        )
        self.bn2 = nn.BatchNorm1d(conv2[0])
        self.conv3 = nn.Conv1d(
            conv2[0], conv3[0], kernel_size=conv3[1], stride=1, padding=conv3[2]
        )
        self.bn3 = nn.BatchNorm1d(conv3[0])
        self.conv4 = nn.Conv1d(
            conv3[0], conv4[0], kernel_size=conv4[1], stride=1, padding=conv4[2]
        )
        self.bn4 = nn.BatchNorm1d(conv4[0])
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0)

        # Compute FC input size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.max_length)
            dummy_output = self.forward_conv(dummy_input)
            num_features_before_fc = dummy_output.numel()

        # create FC layers
        self.fc1 = nn.Linear(num_features_before_fc, fc1_size)
        self.bn5 = nn.BatchNorm1d(fc1_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.bn6 = nn.BatchNorm1d(fc2_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(fc2_size, 1)

    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_conv(self, x: Tensor) -> Tensor:
        """Forward through conv+pool layers."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        return x.view(x.size(0), -1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass. Returns the disruption logit, shape ``(batch, 1)``."""
        x = self.forward_conv(x.unsqueeze(1))
        x = self.dropout1(F.relu(self.bn5(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn6(self.fc2(x))))
        return self.fc3(x)

    def _loss(self, outputs: Tensor, labels: Tensor) -> Tensor:
        """Binary disruption-classification loss."""
        return self._cls_loss(outputs[:, 0], labels[:, 0])

    def _validate_epoch(
        self,
        model: nn.Module,
        dev_loader: DataLoader,
        epoch: int,
        total_train_loss: float,
        train_loader: DataLoader,
        logs: list,
        fbeta: float,
    ) -> None:
        """Run validation for a single epoch and update logs.

        Metrics use the fixed ``DECISION_THRESHOLD`` (default 0.5).
        """
        model.eval()
        total_val_loss = 0.0
        all_classification_targets: list[float] = []
        all_classification_probs: list[float] = []

        with torch.no_grad():
            for data, targets in dev_loader:
                data, targets = data.cuda(), targets.cuda()
                output = model(data)
                classification_targets = targets[:, 0]
                classification_output = output[:, 0]
                classification_probs = torch.sigmoid(classification_output)

                all_classification_targets.extend(classification_targets.cpu().numpy())
                all_classification_probs.extend(classification_probs.cpu().numpy())

                val_total_loss = self._loss(output, targets)
                total_val_loss += val_total_loss.item()

        threshold, val_precision, val_recall = validation_metrics(
            all_classification_targets, all_classification_probs
        )
        all_classification_predictions = [
            prob > threshold for prob in all_classification_probs
        ]

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(dev_loader)
        metrics = {
            THRESHOLD_COL: threshold,
            "Validation Accuracy": accuracy_score(
                all_classification_targets, all_classification_predictions
            ),
            "Validation Precision": val_precision,
            "Validation Recall": val_recall,
            "Validation F1 Score": f1_score(
                all_classification_targets,
                all_classification_predictions,
                zero_division=0,
            ),
            "Validation Fbeta": fbeta_score(
                all_classification_targets,
                all_classification_predictions,
                beta=fbeta,
                zero_division=0,
            ),
        }
        logs.append(
            {
                "epoch": epoch,
                "training_loss": avg_train_loss,
                "validation_loss": avg_val_loss,
                **metrics,
            }
        )

    def train_model(
        self,
        job_id: str,
        lr: float,
        num_epochs: int,
        log_interval: int,
        weight_decay: float,
        lr_scheduler: bool,
        lr_scheduler_factor: float,
        lr_scheduler_patience: int,
        early_stopping_patience: int,
        gradient_clip: float,
        batch_size: int,
        dataloader_num_workers: int,
        fbeta: float = 1.8,
    ) -> None:
        """Train this model on a single device.

        Model selection (best checkpoint + early stopping) maximizes validation
        F-beta among epochs with precision at or above ``MIN_PRECISION``
        (default 0.90) at the fixed decision threshold.
        """
        self.logger.info(f"GPUs Available: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        torch.manual_seed(42)

        lr_scheduler_enabled = lr_scheduler
        num_workers = dataloader_num_workers if torch.cuda.is_available() else 0

        # Log training hyperparameters
        self.logger.info("=" * 60)
        self.logger.info("Training Hyperparameters:")
        self.logger.info(f"  Learning rate: {lr}")
        self.logger.info(f"  Number of epochs: {num_epochs}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Weight decay: {weight_decay}")
        self.logger.info(f"  Log interval: {log_interval}")
        self.logger.info(f"  LR scheduler: {lr_scheduler_enabled}")
        if lr_scheduler_enabled:
            self.logger.info(
                f"    Factor: {lr_scheduler_factor}, Patience: {lr_scheduler_patience}"
            )
        self.logger.info(f"  Early stopping patience: {early_stopping_patience}")
        self.logger.info(f"  Gradient clip: {gradient_clip}")
        self.logger.info(f"  DataLoader num_workers: {num_workers}")
        self.logger.info(
            "  Selection objective: maximize F{:g} with precision >= {:.4f}",
            fbeta,
            min_precision(),
        )
        self.logger.info("=" * 60)

        train, dev, _ = self.dataset.split()

        loader_kw = dict(
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        train_loader = DataLoader(train, shuffle=True, **loader_kw)
        dev_loader = DataLoader(dev, shuffle=False, **loader_kw)

        model = self.cuda()

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if lr_scheduler_enabled:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
            )

        logs = []
        best_score = INFEASIBLE_SCORE
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            if epoch > 0:
                self.logger.info("--------------------------------")

            model.train()
            total_train_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                output = model(data)
                loss_value = self._loss(output, target)

                loss_value.backward()
                if gradient_clip and gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                total_train_loss += loss_value.item()

                if batch_idx % log_interval == 0:
                    dataset_size = len(cast(Sized, train_loader.dataset))
                    self.logger.info(
                        f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}, [{batch_idx * len(data)}/{dataset_size}] Loss {loss_value.item():.6f}"
                    )

            self._validate_epoch(
                model=model,
                dev_loader=dev_loader,
                epoch=epoch,
                total_train_loss=total_train_loss,
                train_loader=train_loader,
                logs=logs,
                fbeta=fbeta,
            )
            avg_val_loss = logs[-1]["validation_loss"] if logs else float("inf")
            last = logs[-1]
            current_score = score(
                float(last[RECALL_COL]),
                float(last[PRECISION_COL]),
                beta=fbeta,
            )

            if lr_scheduler_enabled:
                scheduler.step(avg_val_loss)

            if current_score > best_score:
                best_score = current_score
                epochs_without_improvement = 0
                checkpoint_path = os.path.join(
                    self.prog_dir, f"{job_id}_best_params.pt"
                )
                torch.save(model.state_dict(), checkpoint_path)
                self.logger.info(
                    "New best validation score {:.6f} "
                    "(precision {:.6f}, recall {:.6f}, threshold {:.6f})",
                    best_score,
                    last[PRECISION_COL],
                    last[RECALL_COL],
                    last[THRESHOLD_COL],
                )
            else:
                epochs_without_improvement += 1

            if (
                early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                self.logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs (no improvement for {early_stopping_patience} epochs)"
                )
                break

            if epoch % 5 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(self.prog_dir, f"{job_id}_params_epoch{epoch}.pt"),
                )

        df_logs = pd.DataFrame(logs)
        df_logs.to_csv(
            os.path.join(self.prog_dir, f"{job_id}_training_log.csv"), index=False
        )

        if logs:
            best = best_row(logs)
            self.logger.info("=" * 60)
            self.logger.info(
                "Training complete — best epoch {} (precision {:.6f}, recall {:.6f}, "
                "precision floor {:.4f}):",
                best["epoch"],
                best[PRECISION_COL],
                best[RECALL_COL],
                min_precision(),
            )
            self.logger.info(
                "  Validation Threshold: {:.6f}", best[THRESHOLD_COL]
            )
            self.logger.info("  Validation Recall:    {:.6f}", best[RECALL_COL])
            self.logger.info("  Validation Precision: {:.6f}", best[PRECISION_COL])
            self.logger.info(
                "  Validation F1:        {:.6f}", best["Validation F1 Score"]
            )
            self.logger.info(
                "  Validation F{:g}:        {:.6f}", fbeta, best["Validation Fbeta"]
            )
            self.logger.info(
                "  Selection score:      {:.6f}",
                score(
                    float(best[RECALL_COL]),
                    float(best[PRECISION_COL]),
                    beta=fbeta,
                ),
            )
            self.logger.info("  Validation Loss:      {:.6f}", best["validation_loss"])
            self.logger.info("=" * 60)
