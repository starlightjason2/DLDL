"""Database package: :mod:`database.connection` + :mod:`database.trial_model`."""

from __future__ import annotations

from .connection import (
    Base,
    PathLike,
    SchemaError,
    SCHEMA_USER_VERSION,
    TRIAL_LOG_DB_FILENAME,
    TRIALS_TABLE,
    connect,
    default_trial_db_path,
    initialize,
    sessionmaker_for,
    trial_db_engine,
)
from .trial_model import TRIAL_COLUMN_NAMES, TrialTable

__all__ = [
    "Base",
    "PathLike",
    "SchemaError",
    "TRIAL_COLUMN_NAMES",
    "TRIAL_LOG_DB_FILENAME",
    "TRIALS_TABLE",
    "SCHEMA_USER_VERSION",
    "TrialTable",
    "connect",
    "default_trial_db_path",
    "initialize",
    "sessionmaker_for",
    "trial_db_engine",
]
