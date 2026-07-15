"""Disruption time prediction from smoothed current residuals."""

from __future__ import annotations

import numpy as np

DEFAULT_SMOOTHING = 300

# The boxcar smoother lags the true corner by a fixed fraction of its own window. Measured experimentally.
# LAG = 0.09


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


def get_oriented_current(current):
    return current if np.max(current) > 0.1 else -current


def apply_filter(current):
    oriented = get_oriented_current(current)
    smoothed = apply_smoothing(oriented)

    return oriented - smoothed, smoothed


def predict_disruption_time(current, time) -> float:
    diff, _ = apply_filter(current)
    # dt = float(np.median(np.diff(time)))
    # lag = LAG_WINDOW_FRACTION * get_window_size(current) * dt

    idx_peak = np.argmax(diff)
    idx_trough = np.argmin(diff[idx_peak:]) + idx_peak
    # find where the
    idx_root = np.abs(diff[idx_peak:idx_trough]).argmin()
    # print(lag, dt, get_window_size(current), get_window_size(current) * dt)
    return float(time[int(idx_peak + idx_root)])
