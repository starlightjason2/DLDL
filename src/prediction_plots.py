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
        diff[np.abs(diff) < (3 * sigma)],
        label=f"Best fit Gaussian within $3\\sigma={(3*sigma):2f}$ ms",
    )
    ax.legend()

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = model_dir / "disruption_time_diff.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def generate_scatter_plot(preds, true) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(preds, true, alpha=0.85)

    lo, hi = ax.get_xlim()
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


def main() -> None:
    load_best_model_env()

    df = pd.read_csv(predictions_csv)
    preds = df["pred_root"]

    shots_in_range = df[
        (df["true_time"] < df["pred_end"]) & (df["true_time"] > df["pred_start"])
    ]
    logger.info(
        f"{len(shots_in_range)} / {len(df["true_time"])} shots in range ({len(shots_in_range) / len(df["true_time"])})."
    )

    df["diff"] = preds - df["true_time"]
    generate_scatter_plot(preds, df["true_time"])
    generate_histogram(df)


if __name__ == "__main__":
    main()
