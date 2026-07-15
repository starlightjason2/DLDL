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
from util.disruption_predict import predict_disruption_time
from util.best_model import best_model_dir, load_best_model_cnn, load_best_model_env

_REPO = Path(__file__).resolve().parents[1]
# Env paths are relative to the repo root; run from there so they resolve directly.
os.chdir(_REPO)
load_best_model_env()
data_path = Path(os.environ["DATA_PATH"])
labels_path = Path(os.environ["TRAIN_LABELS_PATH"])
model_dir = best_model_dir()
predictions_csv = model_dir / "predictions.csv"


def _configure_logging(prog_dir: Path) -> None:
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.remove()
    logger.add(sys.stderr, format=fmt, colorize=True, level="INFO")
    logger.add(
        prog_dir / "validate.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
    )


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


def _disruption_time_for_shot(dataset: IpDataset, idx: int) -> tuple[float, float, int]:
    """Predict and record disruption time for one disruptive shot in raw SI time.

    Normalized times are fractions of max_length, mapped back via the raw time column.
    """
    shot = dataset.load_shot_view(idx)
    raw_path = os.path.join(dataset.data_dir, f"{shot.shot_no}.txt")
    raw_current = _read_signal_file(raw_path, col=1)
    raw_time = _read_signal_file(raw_path, col=0)
    max_length = dataset.data.shape[1]
    pred_time = predict_disruption_time(raw_current, raw_time)
    true_time = float(
        raw_time[min(round(shot.t_disrupt * max_length), len(raw_time) - 1)]
    )
    return (shot.index, pred_time, true_time)


def _predict_disruption_times(
    dataset: IpDataset, subset: Subset
) -> list[tuple[int, float, float]]:
    """Compute disruption times for the given holdout shots without running the model."""
    disruption_times: list[tuple[int, float, float]] = []
    for idx in tqdm(list(subset.indices), desc="Predicting times", unit="shot"):
        _, label = dataset[idx]
        if int(label[0].item()) == 1:
            disruption_times.append(_disruption_time_for_shot(dataset, idx))
    return disruption_times


def generate_histogram(df: pd.DataFrame) -> None:
    # Errors in seconds; keep those within +/-10 ms, then convert to microseconds.
    diff = df["diff"][(df["diff"] < 10e-3) & (df["diff"] > -10e-3)] * 1e3
    sigma = diff.std()
    mu = diff.mean()
    logger.success(
        f"Disruption time error (microseconds, n={len(diff)}): "
        f"mean={mu:.3f}, median={diff.median():.3f}, variance={diff.var():.3f}, stddev={sigma:.3f}"
    )
    first_quartile = diff[np.abs(diff) < sigma]
    secondt_quartile = diff[np.abs(diff) < 2 * sigma]
    third_quartile = diff[np.abs(diff) < 3 * sigma]
    logger.success(
        f"{100*len(first_quartile) / len(diff):2f}% shots within 1 stddev, {100*len(secondt_quartile) / len(diff):2f}% shots within 2 stddev, {100*len(third_quartile) / len(diff):2f}% shots within 3 stddev"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(
        diff, bins=75, edgecolor="black", alpha=0.85, range=(-10, 10)
    )
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Difference from real disruption time $\Delta t$ (microseconds)")
    ax.set_ylabel("Count (# shots)")

    # Overlay the best-fit Gaussian. The histogram plots counts, so scale the
    # normal PDF by (n_total * bin_width) to put it on the same vertical axis.
    bin_width = bins[1] - bins[0]
    x = np.linspace(bins[0], bins[-1], 500)
    pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(
        x,
        pdf * len(diff) * bin_width,
        "r--",
        linewidth=1.5,
        label=f"Best fit Gaussian ($\\mu$={mu:.2f}, $\\sigma$={sigma:.2f})",
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


def _write_predictions_csv(
    prog_dir: Path, disruption_times: list[tuple[int, float, float]]
) -> None:
    df = pd.DataFrame(
        [(index, pred, true, true - pred) for index, pred, true in disruption_times],
        columns=["index", "predicted_time", "true_time", "diff"],
    )
    df.sort_values("diff", inplace=True, ascending=False)
    df.to_csv(predictions_csv, index=False)


def _write_artifacts(prog_dir: Path, result: EvalResult) -> None:
    (prog_dir / "false_positives.txt").write_text(
        "\n".join(map(str, sorted(result.fp_shot_ids))) + "\n"
    )
    (prog_dir / "false_negatives.txt").write_text(
        "\n".join(map(str, sorted(result.fn_shot_ids))) + "\n"
    )
    _write_predictions_csv(prog_dir, result.disruption_times)
    logger.info(
        "Wrote misclassified shot ids and {} disruption-time pairs to {}",
        len(result.disruption_times),
        prog_dir,
    )


def _build_dataset() -> IpDataset:
    return IpDataset(
        data_file=os.environ["DATA_PATH"],
        labels_file=os.environ["TRAIN_LABELS_PATH"],
        labels_path=os.environ["LABELS_PATH"],
        data_dir=os.environ["DATA_DIR"],
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )


def predict_disruption_times() -> None:
    """Write predicted/true disruption times for the test holdout without model inference."""
    load_best_model_env()
    dataset = _build_dataset()
    # Evaluate on the test split, held out from model selection (early stopping,
    # best-checkpoint, and threshold tuning all used the dev split during training).
    train, dev, test = dataset.split()

    logger.info(
        f"Predicting disruption times on {len(test)} test holdout shots (no model inference). Model was trained on {len(train)} shots and validated on {len(dev)} shots."
    )
    disruption_times = _predict_disruption_times(dataset, test)
    _write_predictions_csv(model_dir, disruption_times)
    logger.info(
        "Wrote {} disruption-time pairs to {}",
        len(disruption_times),
        predictions_csv,
    )


def evaluate_best_model(batch_size: int = 256) -> None:
    """Run the best_model checkpoint on the test holdout (unseen during training)."""
    load_best_model_env()
    dataset = _build_dataset()

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
    _write_artifacts(model_dir, result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate data and evaluate the best model"
    )
    parser.add_argument(
        "--skip-model-eval",
        action="store_true",
        help="Skip running the best model on the dev holdout set",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help=(
            "Only compute predicted/true disruption times and write predictions.csv; "
            "skip loading and running the model entirely"
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size for model evaluation (default: 256)",
    )
    args = parser.parse_args()

    _configure_logging(model_dir)
    logger.info("Running validation...")
    _require_preprocessed(data_path, labels_path)

    if args.predictions_only:
        predict_disruption_times()
        df = pd.read_csv(predictions_csv)
        generate_scatter_plot(df)
        generate_histogram(df)
    elif not args.skip_model_eval:
        evaluate_best_model(batch_size=args.eval_batch_size)

    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
