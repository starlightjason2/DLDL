"""Disruption time prediction from smoothed current residuals."""

from __future__ import annotations
from enum import IntEnum
import numpy as np

DEFAULT_SMOOTHING = 300


class PredictionType(IntEnum):
    START = 0
    ROOT = 1
    END = 2


def get_window_size(current: np.ndarray):
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


def get_oriented_current(current: np.ndarray):
    return current if np.max(current) > 0.1 else -current


def apply_filter(current: np.ndarray):
    oriented = get_oriented_current(current)
    smoothed = apply_smoothing(oriented)

    return oriented - smoothed, smoothed


def predict_disruption_time(current, time) -> float:
    diff, _ = apply_filter(current)

    idx_peak = np.argmax(diff)
    idx_trough = np.argmin(diff[idx_peak:]) + idx_peak

    # first root after peak
    arr = diff[idx_peak:idx_trough]
    idx_root = next(iter(np.flatnonzero(np.diff(np.signbit(arr)))), 0)
    # idx_root = np.argmin(np.abs(arr))

    return (
        time[idx_peak],
        time[idx_peak + idx_root],
        time[idx_trough],
    )
