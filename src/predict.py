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


@dataclass
class EvalResult:
    y_true: list[int]
    y_pred: list[int]
    fp_shot_ids: list[int]
    fn_shot_ids: list[int]
    disruption_times: list[tuple[float, float, int]]


def _disruption_time_for_shot(
    dataset: IpDataset, idx: int, predictionType=PredictionType.ROOT
) -> tuple[float, float, int]:
    """Predict and record disruption time for one disruptive shot in raw SI time.

    Normalized times are fractions of max_length, mapped back via the raw time column.
    """
    shot = dataset.load_shot_view(idx)
    raw_path = os.path.join(dataset.data_dir, f"{shot.shot_no}.txt")
    raw_current = _read_signal_file(raw_path, col=1)
    raw_time = _read_signal_file(raw_path, col=0)
    max_length = dataset.data.shape[1]
    pred_time = predict_disruption_time(raw_current, raw_time)[predictionType]
    true_time = float(
        raw_time[min(round(shot.t_disrupt * max_length), len(raw_time) - 1)]
    )
    return (shot.index, pred_time, true_time)


def gaussian(x: np.ndarray, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def plot_gaussian(ax, arr, label="Best fit Gaussian"):
    sigma = arr.std()
    mu = arr.mean()

    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 500)
    ax.plot(
        x,
        gaussian(x, mu, sigma),
        linewidth=2,
        label=f"{label} ($\\mu$={mu:.2f}, $\\sigma$={sigma:.2f})",
    )


def generate_histogram(df: pd.DataFrame) -> None:
    # Errors in seconds; keep those within +/-10 ms, then convert to milliseconds.
    diff = df["diff"][(df["diff"] < 10e-3) & (df["diff"] > -10e-3)] * 1e3
    sigma = diff.std()
    mu = diff.mean()
    logger.success(
        f"Disruption time error (milliseconds, n={len(diff)}): "
        f"mean={mu:.3f}, median={diff.median():.3f}, variance={diff.var():.3f}, stddev={sigma:.3f}"
    )
    first_quartile = diff[np.abs(diff) < sigma]
    second_quartile = diff[np.abs(diff) < 2 * sigma]
    third_quartile = diff[np.abs(diff) < 3 * sigma]
    logger.success(
        f"{100*len(first_quartile) / len(diff):2f}% shots within 1 stddev, {100*len(second_quartile) / len(diff):2f}% shots within 2 stddev, {100*len(third_quartile) / len(diff):2f}% shots within 3 stddev"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    start = mu - 5 * sigma
    end = mu + 5 * sigma
    ax.hist(
        diff, bins=60, density=True, edgecolor="black", alpha=0.85, range=(start, end)
    )
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Difference from real disruption time $\Delta t$ (milliseconds)")
    ax.set_ylabel("Count (# shots)")

    # Overlay the best-fit Gaussian
    plot_gaussian(ax, diff)
    plot_gaussian(
        ax,
        diff[np.abs(diff) < 2],
        label="Best fit Gaussian within $\\Delta t \\leq\\pm 2$ ms",
    )
    ax.legend()

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = model_dir / "disruption_time_diff.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def generate_scatter_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["predicted_time"], df["true_time"], alpha=0.85)
    lo = min(df["predicted_time"].min(), df["true_time"].min())
    hi = max(df["predicted_time"].max(), df["true_time"].max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Predicted disruption time (s)")
    ax.set_ylabel("True disruption time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = model_dir / "predictions_scatter.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


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
    disruption_times: list[tuple[float, float, int]] = []
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
                if actual:
                    disruption_times.append(_disruption_time_for_shot(dataset, idx))
            offset += len(preds)
            y_true.extend(actuals.tolist())
            y_pred.extend(preds.tolist())

    return EvalResult(y_true, y_pred, fp_shot_ids, fn_shot_ids, disruption_times)


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
        [(index, pred, true, pred - true) for index, pred, true in disruption_times],
        columns=["index", "predicted_time", "true_time", "diff"],
    )
    df.sort_values("diff", inplace=True, ascending=False)
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

    df = predict_disruption_times(dataset)
    generate_scatter_plot(df)
    generate_histogram(df)

    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
