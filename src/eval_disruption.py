"""Evaluate disruption time predictions on the 20% holdout split."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger

from model.dataset import IpDataset
from util.disruption_predict import predict_disruption_time

_REPO = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_REPO / ".env", encoding="utf-8")


def _abs(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else _REPO / p


def _configure_logging(prog_dir: Path) -> None:
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.remove()
    logger.add(sys.stderr, format=log_format, colorize=True, level="INFO")
    prog_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        prog_dir / "eval_disruption.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
    )


def main() -> None:
    prog_dir = _abs(os.environ["PROG_DIR"])
    _configure_logging(prog_dir)

    dataset = IpDataset(
        normalization_type=os.environ["NORMALIZATION_TYPE"],
        data_file=str(_abs(os.environ["DATA_PATH"])),
        labels_file=str(_abs(os.environ["TRAIN_LABELS_PATH"])),
        labels_path=str(_abs(os.environ["LABELS_PATH"])),
        data_dir=str(_abs(os.environ["DATA_DIR"])),
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )

    train_end = int(0.8 * len(dataset))
    eval_indices = range(train_end, len(dataset))

    disruptive_errors: list[float] = []
    disruptive_predictions: list[tuple[int, float, float]] = []
    false_positive_predictions: list[tuple[int, float]] = []

    for idx in eval_indices:
        shot = dataset.load_shot_view(idx)
        predicted = predict_disruption_time(shot.current)

        if shot.disruptive:
            actual = float(shot.t_disrupt)
            error = abs(predicted - actual)
            disruptive_errors.append(error)
            disruptive_predictions.append((shot.shot_no, predicted, actual))
        else:
            false_positive_predictions.append((shot.shot_no, predicted))

    n_eval = len(eval_indices)
    n_disruptive = len(disruptive_errors)
    n_non_disruptive = len(false_positive_predictions)
    errors_arr = np.array(disruptive_errors)

    logger.info("=" * 60)
    logger.info("Disruption prediction evaluation (20% holdout)")
    logger.info(f"  Eval shots: {n_eval}")
    logger.info(f"  Disruptive: {n_disruptive}")
    logger.info(f"  Non-disruptive: {n_non_disruptive}")
    logger.info("=" * 60)

    if n_disruptive:
        logger.info("Disruptive shots (predicted vs actual normalized time):")
        logger.info(f"  Mean absolute error: {errors_arr.mean():.6f}")
        logger.info(f"  Median absolute error: {np.median(errors_arr):.6f}")
        logger.info(f"  Max absolute error: {errors_arr.max():.6f}")
        logger.info(f"  RMSE: {np.sqrt(np.mean(errors_arr**2)):.6f}")
        within_5pct = 100.0 * np.mean(errors_arr <= 0.05)
        within_10pct = 100.0 * np.mean(errors_arr <= 0.10)
        logger.info(f"  Within 0.05 normalized time: {within_5pct:.1f}%")
        logger.info(f"  Within 0.10 normalized time: {within_10pct:.1f}%")

        worst = max(disruptive_predictions, key=lambda row: abs(row[1] - row[2]))
        logger.info(
            f"  Worst shot {worst[0]}: predicted={worst[1]:.4f}, actual={worst[2]:.4f}"
        )
    else:
        logger.warning("No disruptive shots in holdout set")

    if n_non_disruptive:
        fp_times = np.array([t for _, t in false_positive_predictions])
        logger.info("Non-disruptive shots (method always predicts a time):")
        logger.info(f"  Mean predicted time: {fp_times.mean():.6f}")
        logger.info(f"  Median predicted time: {np.median(fp_times):.6f}")


if __name__ == "__main__":
    main()
