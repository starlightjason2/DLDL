"""Tests for precision-floor / recall objective and threshold tuning."""

from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score

from util.objective import (
    INFEASIBLE_SCORE,
    PRECISION_COL,
    RECALL_COL,
    THRESHOLD_COL,
    best_row,
    best_threshold,
    score,
)


@pytest.fixture(autouse=True)
def _objective_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIN_PRECISION", "0.90")
    monkeypatch.setenv("DECISION_THRESHOLD", "0.5")


def test_best_threshold_prefers_operating_point_over_predict_all() -> None:
    """Regression: must not pick near-zero threshold when 0.5 is feasible."""
    rng = np.random.default_rng(0)
    labels = np.array([0] * 700 + [1] * 300, dtype=int)
    probs = np.where(
        labels == 1,
        rng.uniform(0.55, 0.95, size=labels.size),
        rng.uniform(0.05, 0.45, size=labels.size),
    )

    threshold, precision, recall = best_threshold(labels, probs)

    assert precision >= 0.90
    assert recall > 0.5
    assert 0.01 <= threshold <= 0.99
    assert not (recall == 1.0 and precision < 0.5)


def test_best_threshold_maximizes_recall_under_precision_floor() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)
    probs = np.array([0.10, 0.20, 0.30, 0.85, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96])

    threshold, precision, recall = best_threshold(labels, probs, precision_floor=0.90)

    assert precision >= 0.90
    assert recall == pytest.approx(1.0)
    assert 0.84 < threshold < 0.91


def test_best_threshold_falls_back_to_default_when_infeasible() -> None:
    labels = np.array([0, 0, 0, 1], dtype=int)
    probs = np.array([0.95, 0.94, 0.93, 0.51])

    threshold, precision, recall = best_threshold(labels, probs, precision_floor=0.90)

    assert threshold == 0.5
    assert precision < 0.90
    assert score(recall, precision) == INFEASIBLE_SCORE


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
            THRESHOLD_COL: 0.52,
        },
    ]

    best = best_row(rows)

    assert best["epoch"] == 28
    assert best[RECALL_COL] == pytest.approx(0.913)


def test_metrics_at_default_threshold_match_sklearn() -> None:
    labels = np.array([0, 1, 0, 1, 1], dtype=int)
    probs = np.array([0.2, 0.8, 0.4, 0.7, 0.6])

    _, precision, recall = best_threshold(labels, probs, precision_floor=0.99)
    predictions = probs > 0.5

    assert precision == precision_score(labels, predictions, zero_division=0)
    assert recall == recall_score(labels, predictions, zero_division=0)
