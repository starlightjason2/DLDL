"""Disruption time prediction from smoothed current residuals."""

from __future__ import annotations

from dataclasses import dataclass
from model.dataset import ShotView

import numpy as np
from matplotlib.axes import Axes

DEFAULT_SMOOTHING = 200


def clean_zeros(current: np.ndarray, time: np.ndarray):
    """
    D3D data has a jump to 0 at the end when the shot terminates.
    We want to remove that jump and shift up by the last value to flatten the curve
    """
    # remove trailing zeroes
    processed_current = np.trim_zeros(current).copy()
    processed_time = time[: len(processed_current)].copy()
    return processed_current, processed_time


def apply_smoothing(current: np.ndarray):
    window_size = max(1, len(current) // DEFAULT_SMOOTHING)
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(current, weights, mode="same")
    # smoothing is rough around the edges, so flatten the curve
    smoothed[:window_size] = smoothed[window_size]
    smoothed[-window_size:] = smoothed[-window_size]
    return smoothed


def apply_filter(current, ax: Axes = None):
    smoothed = apply_smoothing(current)
    return np.pow(current - smoothed, 2)


def predict_disruption_time(
    time, current, ax: Axes = None
) -> tuple[np.ndarray, np.ndarray]:
    return float(time[np.argmax(apply_filter(current))])
