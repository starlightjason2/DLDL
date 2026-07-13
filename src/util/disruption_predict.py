"""Disruption time prediction from smoothed current residuals."""

from __future__ import annotations

import numpy as np

DEFAULT_SMOOTHING = 200

# The boxcar smoother lags the true corner by a fixed fraction of its own window. Measured experimentally.
LAG_WINDOW_FRACTION = 0.09


def get_window_size(current):
    return max(1, len(current) // DEFAULT_SMOOTHING)


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
    window_size = get_window_size(current)
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(current, weights, mode="same")
    # smoothing is rough around the edges, so flatten the curve
    edge_value = smoothed[window_size]
    smoothed[:window_size] = edge_value
    smoothed[-window_size:] = edge_value
    return smoothed


def apply_filter(current):
    smoothed = apply_smoothing(current)

    return current - smoothed


def predict_disruption_time(current, time) -> float:
    clean_current, clean_time = clean_zeros(current, time)

    # Orient every shot the same way: plateau above ramp-start → convex corner
    oriented = (
        clean_current if clean_current[-1] >= clean_current[0] else -clean_current
    )

    smoothed = apply_smoothing(oriented)
    residual = oriented - smoothed  # signed, no square

    # Median step is robust to a coarse leading sample in raw time columns.
    dt = float(np.median(np.diff(clean_time)))
    lag = LAG_WINDOW_FRACTION * get_window_size(oriented) * dt
    return float(clean_time[int(np.argmax(residual))]) - lag
