"""Tests for precision-floor / recall objective at a fixed decision threshold."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score

from util.objective import (
    INFEASIBLE_SCORE,
    PRECISION_COL,
    RECALL_COL,
    THRESHOLD_COL,
    best_row,
    score,
    validation_metrics,
)


@pytest.fixture(autouse=True)
def _objective_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIN_PRECISION", "0.90")
    monkeypatch.setenv("DECISION_THRESHOLD", "0.5")


def test_validation_metrics_use_fixed_threshold() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    probs = np.array([0.10, 0.20, 0.40, 0.60, 0.80, 0.90])

    threshold, precision, recall = validation_metrics(labels, probs)

    predictions = probs > 0.5
    assert threshold == 0.5
    assert precision == precision_score(labels, predictions, zero_division=0)
    assert recall == recall_score(labels, predictions, zero_division=0)


def test_validation_metrics_empty_probs() -> None:
    threshold, precision, recall = validation_metrics([], [])

    assert threshold == 0.5
    assert precision == 0.0
    assert recall == 0.0


def test_validation_metrics_no_positive_predictions() -> None:
    labels = np.array([0, 0, 1, 1], dtype=int)
    probs = np.array([0.10, 0.20, 0.30, 0.40])

    threshold, precision, recall = validation_metrics(labels, probs)

    assert threshold == 0.5
    assert precision == 0.0
    assert recall == 0.0


def test_best_row_selects_highest_recall_among_feasible_epochs() -> None:
    rows = [
        {
            "epoch": 13,
            PRECISION_COL: 0.904,
            RECALL_COL: 0.908,
            THRESHOLD_COL: 0.50,
        },
        {
            "epoch": 28,
            PRECISION_COL: 0.901,
            RECALL_COL: 0.913,
            THRESHOLD_COL: 0.50,
        },
    ]

    best = best_row(rows)

    assert best["epoch"] == 28
    assert best[RECALL_COL] == pytest.approx(0.913)


def test_score_requires_precision_floor() -> None:
    assert score(0.95, 0.91) == pytest.approx(0.95)
    assert score(0.99, 0.80) == INFEASIBLE_SCORE
