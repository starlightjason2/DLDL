"""Disruption time prediction from smoothed current residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_SMOOTHING = 100


def predict_disruption_time(
    current: np.ndarray, smoothing: float = DEFAULT_SMOOTHING
) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(len(current)) / len(current)
    window_size = max(1, len(time) // smoothing)
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(current, weights, mode="same")
    diff = np.abs(current - smoothed)
    return float(time[np.argmax(diff)]), diff

