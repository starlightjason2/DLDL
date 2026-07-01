"""Validation objective: recall floor, maximize precision."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

RECALL_COL = "Validation Recall"
PRECISION_COL = "Validation Precision"
INFEASIBLE_SCORE = -1.0


def min_recall() -> float:
    return float(os.environ.get("MIN_RECALL", "0.98"))


def score(recall: float, precision: float, *, recall_floor: float | None = None) -> float:
    floor = min_recall() if recall_floor is None else recall_floor
    return precision if recall > floor else INFEASIBLE_SCORE


def best_row(
    rows: Sequence[Mapping[str, Any]], *, recall_floor: float | None = None
) -> Mapping[str, Any]:
    """Best epoch: max precision among recall-feasible rows, else max recall."""
    if not rows:
        raise ValueError("rows must not be empty")
    floor = min_recall() if recall_floor is None else recall_floor
    feasible = [r for r in rows if float(r[RECALL_COL]) > floor]
    if feasible:
        return max(feasible, key=lambda r: float(r[PRECISION_COL]))
    return max(
        rows,
        key=lambda r: (float(r[RECALL_COL]), float(r[PRECISION_COL])),
    )


def trial_metrics(row: Mapping[str, Any], *, recall_floor: float | None = None) -> dict[str, float]:
    recall = float(row[RECALL_COL])
    precision = float(row[PRECISION_COL])
    return {
        "score": score(recall, precision, recall_floor=recall_floor),
        "recall": recall,
        "precision": precision,
    }
