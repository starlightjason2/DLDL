"""Smoke tests: core packages import without circular-import or missing symbols."""

from __future__ import annotations

import importlib

import pytest

# Library modules only (not train.py / graph.py / validate.py — those need full .env at import time).
_CORE_MODULES = [
    "database.connection",
    "database.tables",
    "database",
    "service.trial_service",
    "schemas",
    "util.data_loading",
    "util.processing",
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


def test_database_exports_engine() -> None:
    from database import Trial, engine, get_db_session

    assert Trial is not None
    assert engine is not None
    assert get_db_session is not None


def test_hptune_mpi_import_optional() -> None:
    """Loads when MPI is available; skipped on laptops without libmpi."""
    try:
        importlib.import_module("hptune_mpi")
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "mpi" in msg or "libmpi" in msg:
            pytest.skip(f"MPI runtime not available: {exc}")
        raise
