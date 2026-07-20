"""Evaluate the best model on the dev holdout set."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

# Prevent macOS OpenMP thread deadlock on CPU inference
if not torch.cuda.is_available():
    torch.set_num_threads(1)

from loguru import logger
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
import pandas as pd

from model.dataset import IpDataset
from util.data_loading import _read_signal_file
from util.disruption_predict import predict_disruption_time, PredictionType
from util.best_model import best_model_dir, load_best_model_cnn, load_best_model_env

_REPO = Path(__file__).resolve().parents[1]
# Env paths are relative to the repo root; run from there so they resolve directly.
os.chdir(_REPO)
load_best_model_env()
data_path = Path(os.environ["DATA_PATH"])
labels_path = Path(os.environ["TRAIN_LABELS_PATH"])
model_dir = best_model_dir()
predictions_csv = model_dir / "predictions.csv"


def _require_preprocessed(data_path: Path, labels_path: Path) -> None:
    missing = [p for p in (data_path, labels_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Preprocessed files not found. Run preprocess_data.py first. "
            f"Missing: {', '.join(map(str, missing))}"
        )
    logger.info("Preprocessed files exist: OK")


def _disruption_time_for_shot(
    dataset: IpDataset, idx: int, predictionType=PredictionType.ROOT
):
    """Predict and record disruption time for one disruptive shot in raw SI time.

    Normalized times are fractions of max_length, mapped back via the raw time column.
    """
    shot = dataset.load_shot_view(idx)
    raw_path = os.path.join(dataset.data_dir, f"{shot.shot_no}.txt")
    raw_current = _read_signal_file(raw_path, col=1)
    raw_time = _read_signal_file(raw_path, col=0)
    max_length = dataset.data.shape[1]
    pred_start, pred_time, pred_end = predict_disruption_time(raw_current, raw_time)
    true_time = float(
        raw_time[min(round(shot.t_disrupt * max_length), len(raw_time) - 1)]
    )
    return (shot.index, true_time, pred_start, pred_time, pred_end)


def predict_disruption_times(dataset: IpDataset) -> None:
    """Write predicted/true disruption times for the test holdout without model inference."""

    # Evaluate on the test split, held out from model selection (early stopping,
    # best-checkpoint, and threshold tuning all used the dev split during training).
    train, dev, test = dataset.split()

    logger.info(
        f"Predicting disruption times on {len(test)} test holdout shots (no model inference). Model was trained on {len(train)} shots and validated on {len(dev)} shots."
    )
    disruption_times: list[tuple[int, float, float]] = []
    for idx in tqdm(list(test.indices), desc="Predicting times", unit="shot"):
        _, label = dataset[idx]
        if int(label[0].item()) == 1:
            disruption_times.append(_disruption_time_for_shot(dataset, idx))

    df = pd.DataFrame(
        disruption_times,
        columns=[
            "index",
            "true_time",
            "pred_start",
            "pred_root",
            "pred_end",
        ],
    )
    df.to_csv(predictions_csv, index=False)
    logger.info(
        f"Wrote {len(disruption_times)} disruption-time pairs to {predictions_csv}",
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate data and evaluate the best model"
    )
    load_best_model_env()
    dataset = IpDataset(
        data_file=os.environ["DATA_PATH"],
        labels_file=os.environ["TRAIN_LABELS_PATH"],
        labels_path=os.environ["LABELS_PATH"],
        data_dir=os.environ["DATA_DIR"],
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )

    predict_disruption_times(dataset)

    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
