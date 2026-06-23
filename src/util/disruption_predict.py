"""Disruption time prediction from smoothed current residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_SMOOTHING = 100


def _smooth_diff(
    current: np.ndarray, smoothing: float = DEFAULT_SMOOTHING
) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(len(current)) / len(current)
    window_size = max(1, len(time) // smoothing)
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(current, weights, mode="same")
    diff = np.abs(current - smoothed)
    return time, diff


def predict_disruption_time(
    current: np.ndarray, smoothing: float = DEFAULT_SMOOTHING
) -> float:
    """Predict disruption time as normalized index of max |current - smoothed|."""
    time, diff = _smooth_diff(current, smoothing)
    return float(time[np.argmax(diff)])


def extract_disruption_features(
    current: np.ndarray, smoothing: float = DEFAULT_SMOOTHING
) -> np.ndarray:
    """Build feature vector from smoothed-residual signal statistics."""
    time, diff = _smooth_diff(current, smoothing)
    peak_idx = int(np.argmax(diff))
    peak_value = float(diff[peak_idx])
    diff_mean = float(diff.mean())
    return np.array(
        [
            time[peak_idx],
            peak_value,
            diff_mean,
            float(diff.std()),
            float(np.percentile(diff, 90)),
            float(np.percentile(diff, 99)),
            float(current.max()),
            float(current.mean()),
            float(current.std()),
            peak_value / (diff_mean + 1e-8),
            peak_idx / len(current),
        ],
        dtype=np.float32,
    )


DISRUPTION_FEATURE_NAMES = (
    "predicted_time",
    "diff_peak",
    "diff_mean",
    "diff_std",
    "diff_p90",
    "diff_p99",
    "current_max",
    "current_mean",
    "current_std",
    "peak_to_mean_ratio",
    "peak_index_norm",
)
