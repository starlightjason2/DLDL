"""Disruption time prediction from smoothed current residuals."""

from __future__ import annotations

from dataclasses import dataclass
from model.dataset import ShotView

import numpy as np
from matplotlib.axes import Axes

DEFAULT_SMOOTHING = 200


def remove_jump_to_zero(current: np.ndarray):
    """
    D3D data has a jump to 0 at the end when the shot terminates.
    We want to remove that jump and shift up by the last value to flatten the curve
    """
    # remove trailing zeroes
    processed_current = np.trim_zeros(np.copy(current))
    # shift up
    processed_current -= processed_current[-1]
    # pad with zeroes to match the original data shape
    return np.pad(processed_current, (0, len(current) - len(processed_current)))


def apply_smoothing(current: np.ndarray):
    window_size = max(1, len(current) // DEFAULT_SMOOTHING)
    weights = np.ones(window_size) / window_size
    return np.convolve(current, weights, mode="same")


def apply_filter(shot: ShotView, ax: Axes = None):
    processed_current = remove_jump_to_zero(shot.current)
    smoothed = apply_smoothing(processed_current)
    filtered_current = np.abs(shot.current - smoothed)

    # the algorithm is weird around the edges, so throw edge values out
    filtered_current[:2] = 0
    filtered_current[-2:] = 0

    return filtered_current


def predict_disruption_time(
    shot: ShotView, ax: Axes = None
) -> tuple[np.ndarray, np.ndarray]:
    return float(shot.time[np.argmax(apply_filter(shot))])
