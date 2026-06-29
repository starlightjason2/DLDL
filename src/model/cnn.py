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
    precision_score,
    recall_score,
)
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.dataset import IpDataset


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
        self.logger.info(f"  Normalization type: {dataset.normalization_type}")
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
        writer: SummaryWriter,
        total_train_loss: float,
        train_loader: DataLoader,
        logs: list,
        fbeta: float,
    ) -> None:
        """Run validation for a single epoch and update logs.

        ``fbeta`` is the beta used for the F-beta score that drives model
        selection (beta > 1 weights recall over precision).
        """
        model.eval()
        total_val_loss = 0.0
        all_classification_targets, all_classification_predictions = [], []

        with torch.no_grad():
            for data, targets in dev_loader:
                data, targets = data.cuda(), targets.cuda()
                output = model(data)
                classification_targets = targets[:, 0]
                classification_output = output[:, 0]

                classification_predictions = (
                    torch.sigmoid(classification_output) > self.decision_threshold
                )
                all_classification_targets.extend(classification_targets.cpu().numpy())
                all_classification_predictions.extend(
                    classification_predictions.cpu().numpy()
                )

                val_total_loss = self._loss(output, targets)
                total_val_loss += val_total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(dev_loader)
        metrics = {
            "Validation Accuracy": accuracy_score(
                all_classification_targets, all_classification_predictions
            ),
            "Validation Precision": precision_score(
                all_classification_targets,
                all_classification_predictions,
                zero_division=0,
            ),
            "Validation Recall": recall_score(
                all_classification_targets,
                all_classification_predictions,
                zero_division=0,
            ),
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
        for name, value in metrics.items():
            writer.add_scalar(name, value, epoch)
        writer.add_scalar("Validation Loss", avg_val_loss, epoch)

    @staticmethod
    def _checkpoint(
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        best_score: float,
        epochs_without_improvement: int,
        fbeta: float,
    ) -> dict:
        """Full-state checkpoint: weights plus everything needed to continue training."""
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "fbeta": fbeta,
        }

    def _warm_start(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        *,
        lr: float,
        weight_decay: float,
        chained: bool = False,
    ) -> None:
        """Initialize weights (and optimizer momentum) from a prior checkpoint.

        Used to pick up off the best trial found so far. Only the model weights and
        optimizer state transfer; this trial's own ``lr``/``weight_decay`` are kept,
        and its epoch counter and early-stopping state start fresh so each trial
        remains an independent, comparable evaluation for the tuner.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Tolerate both full-state checkpoints and bare weight state_dicts.
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)

        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["weight_decay"] = weight_decay

        source_epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
        source_score = ckpt.get("best_score") if isinstance(ckpt, dict) else None
        verb = "Chained" if chained else "Warm-started"
        self.logger.info(
            "{} from {} (source epoch={}, source best F-beta={})",
            verb,
            checkpoint_path,
            source_epoch,
            f"{source_score:.6f}" if isinstance(source_score, (int, float)) else "n/a",
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
        warm_start_checkpoint: str | None = None,
        chained: bool = False,
    ) -> None:
        """Train this model on a single device.

        Model selection (best checkpoint + early stopping) maximizes the
        validation F-beta score, where ``fbeta`` > 1 prioritizes recall over
        precision. The same F-beta is the scalar objective reported to the
        hyperparameter tuner.
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
        self.logger.info(f"  Selection objective: F{fbeta:g} score (maximize)")
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

        if warm_start_checkpoint:
            self._warm_start(
                warm_start_checkpoint,
                model,
                optimizer,
                lr=lr,
                weight_decay=weight_decay,
                chained=chained,
            )

        logs = []
        writer = SummaryWriter(self.prog_dir, filename_suffix=f"-job_{job_id}")

        best_score = float("-inf")
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
                    step = epoch * dataset_size + batch_idx
                    self.logger.info(
                        f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}, [{batch_idx * len(data)}/{dataset_size}] Loss {loss_value.item():.6f}"
                    )
                    writer.add_scalar("Training Loss", loss_value.item(), step)

            self._validate_epoch(
                model=model,
                dev_loader=dev_loader,
                epoch=epoch,
                writer=writer,
                total_train_loss=total_train_loss,
                train_loader=train_loader,
                logs=logs,
                fbeta=fbeta,
            )
            avg_val_loss = logs[-1]["validation_loss"] if logs else float("inf")
            current_score = logs[-1]["Validation Fbeta"] if logs else float("-inf")

            if lr_scheduler_enabled:
                scheduler.step(avg_val_loss)
                writer.add_scalar(
                    "Learning Rate", optimizer.param_groups[0]["lr"], epoch
                )

            if current_score > best_score:
                best_score = current_score
                epochs_without_improvement = 0
                torch.save(
                    self._checkpoint(
                        model,
                        optimizer,
                        epoch,
                        best_score,
                        epochs_without_improvement,
                        fbeta,
                    ),
                    os.path.join(self.prog_dir, f"{job_id}_best_params.pt"),
                )
                self.logger.info(f"New best validation F{fbeta:g}: {best_score:.6f}")
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
                    self._checkpoint(
                        model,
                        optimizer,
                        epoch,
                        best_score,
                        epochs_without_improvement,
                        fbeta,
                    ),
                    os.path.join(self.prog_dir, f"{job_id}_params_epoch{epoch}.pt"),
                )

        writer.close()
        df_logs = pd.DataFrame(logs)
        df_logs.to_csv(
            os.path.join(self.prog_dir, f"{job_id}_training_log.csv"), index=False
        )

        if logs:
            best = max(logs, key=lambda row: row["Validation Fbeta"])
            self.logger.info("=" * 60)
            self.logger.info(
                "Training complete — best epoch {} (selected by F{:g}):",
                best["epoch"],
                fbeta,
            )
            self.logger.info("  Validation Recall:    {:.6f}", best["Validation Recall"])
            self.logger.info(
                "  Validation Precision: {:.6f}", best["Validation Precision"]
            )
            self.logger.info("  Validation F1:        {:.6f}", best["Validation F1 Score"])
            self.logger.info(
                "  Validation F{:g}:        {:.6f}", fbeta, best["Validation Fbeta"]
            )
            self.logger.info("  Validation Loss:      {:.6f}", best["validation_loss"])
            self.logger.info("=" * 60)
