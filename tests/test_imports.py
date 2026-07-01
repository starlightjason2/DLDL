"""Smoke tests: core packages import without circular-import or missing symbols."""

from __future__ import annotations

import importlib

import pytest

# Library modules only (not train.py / graph.py / validate.py — those need full .env at import time).
_CORE_MODULES = [
    "service.trial_service",
    "schemas",
    "util.data_loading",
    "util.processing",
    "util.objective",
    "util.hptune",
    "model.dataset",
    "model.bayesian_hptuner",
    "model.hp_trial",
    "hptune_serial",
    "model.cnn",
]


@pytest.mark.parametrize("name", _CORE_MODULES)
def test_import_core_module(name: str) -> None:
    importlib.import_module(name)
