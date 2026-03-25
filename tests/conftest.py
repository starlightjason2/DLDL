"""Pytest configuration: env vars before :mod:`database.connection` loads."""

from __future__ import annotations

import os

os.environ.setdefault(
    "DB_CONNECTION",
    "sqlite:////tmp/dldl_pytest_imports.db",
)
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
