"""Validation objective: precision floor, maximize recall."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import precision_score, recall_score

RECALL_COL = "Validation Recall"
PRECISION_COL = "Validation Precision"
THRESHOLD_COL = "Validation Threshold"
INFEASIBLE_SCORE = -1.0


def min_precision() -> float:
    return float(os.environ.get("MIN_PRECISION", "0.90"))


def score(
    recall: float, precision: float, *, precision_floor: float | None = None
) -> float:
    floor = min_precision() if precision_floor is None else precision_floor
    return recall if precision >= floor else INFEASIBLE_SCORE


def _threshold_candidates(probs: np.ndarray) -> np.ndarray:
    """Candidate cutoffs: uniform grid plus every distinct predicted probability."""
    grid = np.linspace(0.01, 0.99, 99)
    return np.unique(np.concatenate([probs, grid]))


def _metrics_at_threshold(
    labels: np.ndarray, probs: np.ndarray, threshold: float
) -> tuple[float, float]:
    predictions = probs > threshold
    if not predictions.any():
        return 0.0, 0.0
    return (
        precision_score(labels, predictions, zero_division=0),
        recall_score(labels, predictions, zero_division=0),
    )


def default_threshold() -> float:
    return float(os.environ.get("DECISION_THRESHOLD", "0.5"))


def best_threshold(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    precision_floor: float | None = None,
) -> tuple[float, float, float]:
    """Return ``(threshold, precision, recall)`` maximizing recall with P >= floor."""
    floor = min_precision() if precision_floor is None else precision_floor
    labels = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)

    if probs.size == 0:
        return default_threshold(), 0.0, INFEASIBLE_SCORE

    best_t, best_p, best_r = default_threshold(), 0.0, INFEASIBLE_SCORE
    for threshold in _threshold_candidates(probs):
        precision, recall = _metrics_at_threshold(labels, probs, float(threshold))
        candidate_score = score(recall, precision, precision_floor=floor)
        if candidate_score <= INFEASIBLE_SCORE:
            continue
        if candidate_score > best_r or (
            candidate_score == best_r and precision > best_p
        ):
            best_t, best_p, best_r = float(threshold), precision, recall

    if best_r > INFEASIBLE_SCORE:
        return best_t, best_p, best_r

    fallback_t = default_threshold()
    fallback_p, fallback_r = _metrics_at_threshold(labels, probs, fallback_t)
    return fallback_t, fallback_p, fallback_r


def best_row(
    rows: Sequence[Mapping[str, Any]], *, precision_floor: float | None = None
) -> Mapping[str, Any]:
    """Best epoch: max recall among precision-feasible rows, else max precision."""
    if not rows:
        raise ValueError("rows must not be empty")
    floor = min_precision() if precision_floor is None else precision_floor
    feasible = [r for r in rows if float(r[PRECISION_COL]) >= floor]
    if feasible:
        return max(feasible, key=lambda r: float(r[RECALL_COL]))
    return max(
        rows,
        key=lambda r: (float(r[PRECISION_COL]), float(r[RECALL_COL])),
    )


def trial_metrics(
    row: Mapping[str, Any], *, precision_floor: float | None = None
) -> dict[str, float]:
    recall = float(row[RECALL_COL])
    precision = float(row[PRECISION_COL])
    return {
        "score": score(recall, precision, precision_floor=precision_floor),
        "recall": recall,
        "precision": precision,
    }
