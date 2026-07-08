"""Smoke tests for the F-beta objective (pure numpy/sklearn; no training)."""

from __future__ import annotations

import numpy as np
import pytest

from util.objective import (
    INFEASIBLE_SCORE,
    F1_COL,
    PRECISION_COL,
    RECALL_COL,
    best_row,
    fbeta_from_pr,
    score,
    trial_metrics,
    validation_metrics,
)


@pytest.fixture(autouse=True)
def _objective_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIN_PRECISION", "0.90")
    monkeypatch.setenv("DECISION_THRESHOLD", "0.5")
    monkeypatch.setenv("FBETA", "1.8")


def test_validation_metrics_at_fixed_threshold() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    probs = np.array([0.10, 0.20, 0.40, 0.60, 0.80, 0.90])

    threshold, precision, recall = validation_metrics(labels, probs)

    assert threshold == 0.5
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_score_respects_precision_floor() -> None:
    assert score(0.95, 0.91) == pytest.approx(fbeta_from_pr(0.95, 0.91))
    assert score(0.99, 0.80) == INFEASIBLE_SCORE


def test_best_row_picks_highest_feasible_fbeta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FBETA", "1.0")

    rows = [
        {"epoch": 13, PRECISION_COL: 0.950, RECALL_COL: 0.880, "Validation Threshold": 0.5},
        {"epoch": 28, PRECISION_COL: 0.901, RECALL_COL: 0.913, "Validation Threshold": 0.5},
    ]

    best = best_row(rows)

    assert best["epoch"] == 13


def test_trial_metrics_from_log_row() -> None:
    metrics = trial_metrics(
        {PRECISION_COL: 0.91, RECALL_COL: 0.95, F1_COL: 0.929}
    )

    assert metrics["f1"] == pytest.approx(0.929)
    assert metrics["score"] == pytest.approx(fbeta_from_pr(0.95, 0.91))
