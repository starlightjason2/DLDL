"""Smoke tests for HP tune search spaces (env parsing only; no training)."""

from __future__ import annotations

import pytest

from model.hp_trial import HpTuneTrial
from model.hyperparam_space import ArchitectureHyperparameterSpace, HyperparameterSpace


def test_training_hyperparam_space_reads_env() -> None:
    space = HyperparameterSpace.from_env()

    assert space.allowed_epochs == (2, 4)
    assert space.batch_sizes == (4, 8)
    assert "lr" in space.bounds


def test_architecture_space_derives_same_padding() -> None:
    space = ArchitectureHyperparameterSpace.from_env()
    proposal = space.suggestion_to_trial(
        {
            "conv1_f_idx": 1.0,
            "conv2_f_idx": 0.0,
            "conv3_f_idx": 1.0,
            "conv4_f_idx": 0.0,
            "conv1_k_idx": 1.0,
            "conv2_k_idx": 0.0,
            "conv3_k_idx": 1.0,
            "conv4_k_idx": 0.0,
            "pool_idx": 0.0,
            "fc1": 12.0,
            "fc2": 6.0,
        }
    )

    assert proposal["conv1_kernel"] == 5
    assert proposal["conv1_padding"] == 2
    assert proposal["fc2_size"] <= proposal["fc1_size"]


def test_architecture_trial_signature_differs_from_training() -> None:
    training = HpTuneTrial.proposed_signature(
        {
            "lr": 0.001,
            "epochs": 4,
            "dropout": 0.1,
            "weight_decay": 1e-5,
            "batch_size": 4,
            "gradient_clip": 1.0,
            "lr_scheduler": False,
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 5,
            "early_stopping_patience": 3,
            "cls_pos_weight": 1.0,
        }
    )
    architecture = HpTuneTrial.proposed_signature(
        {
            "lr": 0.001,
            "epochs": 4,
            "dropout": 0.1,
            "weight_decay": 1e-5,
            "batch_size": 4,
            "gradient_clip": 1.0,
            "lr_scheduler": False,
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 5,
            "early_stopping_patience": 3,
            "cls_pos_weight": 1.0,
            "conv1_filters": 16,
            "conv1_kernel": 5,
            "conv1_padding": 2,
            "conv2_filters": 8,
            "conv2_kernel": 3,
            "conv2_padding": 1,
            "conv3_filters": 16,
            "conv3_kernel": 5,
            "conv3_padding": 2,
            "conv4_filters": 8,
            "conv4_kernel": 3,
            "conv4_padding": 1,
            "pool_size": 2,
            "fc1_size": 12,
            "fc2_size": 6,
        }
    )

    assert training != architecture
