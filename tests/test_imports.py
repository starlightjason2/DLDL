"""Smoke tests: core modules import without circular-import errors."""

from __future__ import annotations

import importlib

import pytest

# Library modules only. Entry scripts (train.py, validate.py, graph.py) need a full
# runtime .env and are not imported here.
_CORE_MODULES = [
    "util.objective",
    "util.data_loading",
    "util.processing",
    "util.training",
    "util.best_model",
    "util.hp_tune",
    "model.trial_status",
    "model.hyperparam_space",
    "model.hp_trial",
    "model.dataset",
    "model.cnn",
    "service.trial_service",
    "model.bayesian_hp_tuner",
    "hp_tune_serial",
]


@pytest.mark.parametrize("name", _CORE_MODULES)
def test_import_core_module(name: str) -> None:
    importlib.import_module(name)
