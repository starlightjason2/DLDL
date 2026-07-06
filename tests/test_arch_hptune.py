"""Tests for architecture HPTune search space and trial wiring."""

from __future__ import annotations

import os

import pytest

from model.hp_trial import HPTuneTrial
from model.hyperparam_space import ArchitectureHyperparameterSpace


@pytest.fixture
def arch_hptune_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCH_HPTUNE_CONV_FILTERS", "8,16,32")
    monkeypatch.setenv("ARCH_HPTUNE_KERNELS", "3,5,9")
    monkeypatch.setenv("ARCH_HPTUNE_POOL_SIZES", "2,4")
    monkeypatch.setenv("ARCH_HPTUNE_FC1_MIN", "64")
    monkeypatch.setenv("ARCH_HPTUNE_FC1_MAX", "256")
    monkeypatch.setenv("ARCH_HPTUNE_FC2_MIN", "32")
    monkeypatch.setenv("ARCH_HPTUNE_FC2_MAX", "128")
    monkeypatch.setenv("ARCH_HPTUNE_NUM_INITIAL_TRIALS", "5")
    monkeypatch.setenv("ARCH_HPTUNE_RANDOM_INSERT_EVERY", "3")
    monkeypatch.setenv("ARCH_HPTUNE_EI_XI", "0.05")


def test_architecture_space_derives_same_padding(
    arch_hptune_env: None,
) -> None:
    space = ArchitectureHyperparameterSpace.from_env()
    proposal = space.suggestion_to_trial(
        {
            "conv1_f_idx": 2.0,
            "conv2_f_idx": 1.0,
            "conv3_f_idx": 0.0,
            "conv4_f_idx": 2.0,
            "conv1_k_idx": 2.0,
            "conv2_k_idx": 1.0,
            "conv3_k_idx": 0.0,
            "conv4_k_idx": 2.0,
            "pool_idx": 1.0,
            "fc1": 120.0,
            "fc2": 60.0,
        }
    )

    assert proposal["conv1_kernel"] == 9
    assert proposal["conv1_padding"] == 4
    assert proposal["fc2_size"] <= proposal["fc1_size"]


def test_architecture_trial_signature_differs_from_training() -> None:
    training = HPTuneTrial.proposed_signature(
        {
            "lr": 0.001,
            "epochs": 100,
            "dropout": 0.5,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "gradient_clip": 1.0,
            "lr_scheduler": True,
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 5,
            "early_stopping_patience": 10,
            "cls_pos_weight": 2.0,
        }
    )
    architecture = HPTuneTrial.proposed_signature(
        {
            "lr": 0.001,
            "epochs": 100,
            "dropout": 0.5,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "gradient_clip": 1.0,
            "lr_scheduler": True,
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 5,
            "early_stopping_patience": 10,
            "cls_pos_weight": 2.0,
            "conv1_filters": 64,
            "conv1_kernel": 7,
            "conv1_padding": 3,
            "conv2_filters": 128,
            "conv2_kernel": 7,
            "conv2_padding": 3,
            "conv3_filters": 32,
            "conv3_kernel": 5,
            "conv3_padding": 2,
            "conv4_filters": 64,
            "conv4_kernel": 7,
            "conv4_padding": 3,
            "pool_size": 8,
            "fc1_size": 72,
            "fc2_size": 72,
        }
    )

    assert training != architecture
