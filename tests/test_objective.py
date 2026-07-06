"""Tests for F-beta objective with a precision floor."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import fbeta_score, precision_score, recall_score

from util.objective import (
    INFEASIBLE_SCORE,
    F1_COL,
    PRECISION_COL,
    RECALL_COL,
    THRESHOLD_COL,
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


def test_fbeta_from_pr_matches_sklearn() -> None:
    labels = np.array([0, 0, 1, 1, 1], dtype=int)
    probs = np.array([0.2, 0.3, 0.7, 0.8, 0.9])
    predictions = probs > 0.5
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)

    assert fbeta_from_pr(recall, precision, beta=1.8) == pytest.approx(
        fbeta_score(labels, predictions, beta=1.8, zero_division=0)
    )


def test_score_uses_fbeta_with_precision_floor() -> None:
    recall, precision = 0.95, 0.91
    assert score(recall, precision) == pytest.approx(fbeta_from_pr(recall, precision))
    assert score(0.99, 0.80) == INFEASIBLE_SCORE


def test_trial_metrics_includes_f1_from_training_log() -> None:
    row = {
        PRECISION_COL: 0.91,
        RECALL_COL: 0.95,
        F1_COL: 0.929,
    }

    metrics = trial_metrics(row)

    assert metrics["f1"] == pytest.approx(0.929)
    assert metrics["score"] == pytest.approx(fbeta_from_pr(0.95, 0.91))


def test_best_row_selects_highest_fbeta_among_feasible_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FBETA", "1.0")

    rows = [
        {
            "epoch": 13,
            PRECISION_COL: 0.950,
            RECALL_COL: 0.880,
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

    assert best["epoch"] == 13
    assert best[PRECISION_COL] == pytest.approx(0.950)


def test_best_row_prefers_higher_recall_when_fbeta_tied_by_beta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FBETA", "1.8")

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
