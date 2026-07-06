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


def default_threshold() -> float:
    return float(os.environ.get("DECISION_THRESHOLD", "0.5"))


def validation_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> tuple[float, float, float]:
    """Return ``(threshold, precision, recall)`` at the fixed decision threshold."""
    threshold = default_threshold()
    labels = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)
    if probs.size == 0:
        return threshold, 0.0, 0.0

    predictions = probs > threshold
    if not predictions.any():
        return threshold, 0.0, 0.0

    return (
        threshold,
        precision_score(labels, predictions, zero_division=0),
        recall_score(labels, predictions, zero_division=0),
    )


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
