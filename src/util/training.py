"""Training helpers: build a CNN from env vars and load checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from model.cnn import IpCNN
    from model.dataset import IpDataset


def build_cnn_from_env(dataset: "IpDataset", prog_dir: str) -> "IpCNN":
    from model.cnn import IpCNN

    def _conv(prefix: str) -> tuple[int, int, int]:
        return (
            int(os.environ[f"{prefix}_FILTERS"]),
            int(os.environ[f"{prefix}_KERNEL"]),
            int(os.environ[f"{prefix}_PADDING"]),
        )

    return IpCNN(
        dataset,
        prog_dir=prog_dir,
        conv1=_conv("CONV1"),
        conv2=_conv("CONV2"),
        conv3=_conv("CONV3"),
        conv4=_conv("CONV4"),
        pool_size=int(os.environ["POOL_SIZE"]),
        fc1_size=int(os.environ["FC1_SIZE"]),
        fc2_size=int(os.environ["FC2_SIZE"]),
        dropout_rate=float(os.environ["DROPOUT_RATE"]),
        cls_pos_weight=float(os.environ["CLS_POS_WEIGHT"]),
        decision_threshold=float(os.environ["DECISION_THRESHOLD"]),
    )


def load_checkpoint_into_model(model: "IpCNN", checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
