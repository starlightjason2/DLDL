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


def _require_preprocessed(data_path: Path, labels_path: Path) -> None:
    missing = [p for p in (data_path, labels_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Preprocessed files not found. Run preprocess_data.py first. "
            f"Missing: {', '.join(map(str, missing))}"
        )
    logger.info("Preprocessed files exist: OK")


@dataclass
class EvalResult:
    y_true: list[int]
    y_pred: list[int]
    fp_shot_ids: list[int]
    fn_shot_ids: list[int]


def _disruption_time_for_shot(
    dataset: IpDataset, idx: int, predictionType=PredictionType.ROOT
) -> tuple[int, float]:
    """Predict and record disruption time for one disruptive shot in raw SI time.

    Normalized times are fractions of max_length, mapped back via the raw time column.
    """
    shot = dataset.load_shot_view(idx)
    raw_path = os.path.join(dataset.data_dir, f"{shot.shot_no}.txt")
    raw_current = _read_signal_file(raw_path, col=1)
    raw_time = _read_signal_file(raw_path, col=0)
    max_length = dataset.data.shape[1]
    true_time = float(
        raw_time[min(round(shot.t_disrupt * max_length), len(raw_time) - 1)]
    )
    return (shot.index, true_time)


def _evaluate_split(
    dataset: IpDataset,
    model: torch.nn.Module,
    subset: Subset,
    *,
    batch_size: int,
    device: str,
) -> EvalResult:
    subset_indices = list(subset.indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    threshold = model.decision_threshold
    y_true: list[int] = []
    y_pred: list[int] = []
    fp_shot_ids: list[int] = []
    fn_shot_ids: list[int] = []
    offset = 0

    with torch.no_grad():
        for signals, labels in tqdm(loader, desc="Evaluating", unit="batch"):
            probs = torch.sigmoid(model(signals.float().to(device))[:, 0]).cpu().numpy()
            preds = (probs > threshold).astype(int)
            actuals = labels[:, 0].cpu().numpy().astype(int)

            for i, (pred, actual) in enumerate(zip(preds, actuals)):
                idx = subset_indices[offset + i]
                if pred != actual:
                    (fp_shot_ids if pred else fn_shot_ids).append(idx)

            offset += len(preds)
            y_true.extend(actuals.tolist())
            y_pred.extend(preds.tolist())

    return EvalResult(y_true, y_pred, fp_shot_ids, fn_shot_ids)


def _log_metrics(result: EvalResult, *, fbeta: float) -> None:
    tn, fp, fn, tp = confusion_matrix(
        result.y_true, result.y_pred, labels=[0, 1]
    ).ravel()
    logger.info("=" * 60)
    logger.info("Best-model evaluation ({} test holdout shots):", len(result.y_true))
    logger.info("  Accuracy:  {:.6f}", accuracy_score(result.y_true, result.y_pred))
    logger.info(
        "  Precision: {:.6f}",
        precision_score(result.y_true, result.y_pred, zero_division=0),
    )
    logger.info(
        "  Recall:    {:.6f}",
        recall_score(result.y_true, result.y_pred, zero_division=0),
    )
    logger.info(
        "  F1:        {:.6f}",
        f1_score(result.y_true, result.y_pred, zero_division=0),
    )
    logger.info(
        "  F{:g}:        {:.6f}",
        fbeta,
        fbeta_score(result.y_true, result.y_pred, beta=fbeta, zero_division=0),
    )
    logger.info("  Confusion: TP={} FP={} FN={} TN={}", tp, fp, fn, tn)
    logger.info(
        "  False positives: {} | False negatives: {}",
        len(result.fp_shot_ids),
        len(result.fn_shot_ids),
    )
    logger.info("  False-positive shot ids: {}", sorted(result.fp_shot_ids))
    logger.info("  False-negative shot ids: {}", sorted(result.fn_shot_ids))
    logger.info("=" * 60)


def evaluate_best_model(dataset: IpDataset, batch_size: int = 256) -> None:
    """Run the best_model checkpoint on the test holdout (unseen during training)."""

    # Evaluate on the test split, held out from model selection (early stopping,
    # best-checkpoint, and threshold tuning all used the dev split during training).
    _, _, test = dataset.split()

    model = load_best_model_cnn(dataset)
    if model is None:
        logger.warning(
            "No checkpoint found in best_model/*_best_params.pt; skipping model evaluation."
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        batch_size = min(batch_size, 8)  # 60k-length sequences are huge on CPU
    model = model.to(device)
    fbeta = float(os.environ.get("FBETA", "1.8"))

    logger.info(
        "Evaluating best model on {} test holdout shots (device={}, threshold={:.4f})",
        len(test),
        device,
        model.decision_threshold,
    )

    result = _evaluate_split(dataset, model, test, batch_size=batch_size, device=device)
    _log_metrics(result, fbeta=fbeta)

    (model_dir / "false_positives.txt").write_text(
        "\n".join(map(str, sorted(result.fp_shot_ids))) + "\n"
    )
    (model_dir / "false_negatives.txt").write_text(
        "\n".join(map(str, sorted(result.fn_shot_ids))) + "\n"
    )

    logger.info(
        f"Wrote misclassified shot ids to {model_dir}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate data and evaluate the best model"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size for model evaluation (default: 256)",
    )
    args = parser.parse_args()

    _require_preprocessed(data_path, labels_path)

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

    evaluate_best_model(dataset, batch_size=args.eval_batch_size)


if __name__ == "__main__":
    main()
