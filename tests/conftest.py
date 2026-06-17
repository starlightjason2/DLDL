"""Pytest configuration: env vars before core modules load."""

from __future__ import annotations

import os

os.environ.setdefault("HPTUNE_DIR", "/tmp/dldl_pytest_hptune")
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
