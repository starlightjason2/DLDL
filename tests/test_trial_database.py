"""Tests for parallel HP-tuning SQLite trial log (SQLAlchemy ORM)."""

import os
import sqlite3
import tempfile

import pytest

from database import (
    SCHEMA_USER_VERSION,
    TRIAL_COLUMN_NAMES,
    TRIAL_LOG_DB_FILENAME,
    TRIALS_TABLE,
    SchemaError,
    connect,
    default_trial_db_path,
    initialize,
    trial_db_engine,
)
from model.hptune_trial import HPTuneTrial
from service.trial_service import TrialService
from util.hptune import TRIAL_LOG_COLUMNS


def test_schema_column_names_match_csv_columns():
    assert list(TRIAL_COLUMN_NAMES) == TRIAL_LOG_COLUMNS


def test_initialize_creates_file_and_tables():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        assert not os.path.isfile(db_path)
        initialize(db_path)
        assert os.path.isfile(db_path)
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute("PRAGMA user_version")
            assert cur.fetchone()[0] == SCHEMA_USER_VERSION
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (TRIALS_TABLE,),
            )
            assert cur.fetchone() is not None
            cur = conn.execute(f'PRAGMA table_info("{TRIALS_TABLE}")')
            cols = [row[1] for row in cur.fetchall()]
            assert cols == TRIAL_LOG_COLUMNS
        finally:
            conn.close()


def test_initialize_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        initialize(db_path)
        initialize(db_path)
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute("PRAGMA user_version")
            assert cur.fetchone()[0] == SCHEMA_USER_VERSION
        finally:
            conn.close()


def test_default_path():
    with tempfile.TemporaryDirectory() as tmp:
        p = default_trial_db_path(tmp)
        assert p == os.path.join(tmp, TRIAL_LOG_DB_FILENAME)


def test_connect_creates_parent_dir():
    with tempfile.TemporaryDirectory() as tmp:
        nested = os.path.join(tmp, "a", "b", "trials_log.db")
        conn = connect(nested)
        try:
            assert os.path.isdir(os.path.dirname(nested))
        finally:
            conn.close()


def test_service_persist_snapshot_get_trials_roundtrip():
    """Bulk replace + list lives on :class:`service.trial_service.TrialService`."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        svc = TrialService(db_path)
        trial = HPTuneTrial(
            lr=1e-3,
            epochs=10,
            dropout=0.2,
            weight_decay=1e-4,
            batch_size=32,
            gradient_clip=1.0,
            lr_scheduler=True,
            lr_scheduler_factor=0.5,
            lr_scheduler_patience=2,
            early_stopping_patience=3,
            trial_id="trial_1",
            val_loss=0.5,
            status=0,
            retries=0,
        )
        svc.persist_snapshot([trial])
        out = svc.get_trials()
        assert len(out) == 1
        assert out[0].trial_id == "trial_1"
        assert out[0].val_loss == pytest.approx(0.5)


def test_initialize_rejects_newer_user_version():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(f"PRAGMA user_version = {SCHEMA_USER_VERSION + 9}")
            conn.commit()
        finally:
            conn.close()
        with pytest.raises(SchemaError):
            initialize(db_path)


def test_trial_db_engine_returns_engine():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        eng = trial_db_engine(db_path)
        assert "trials_log.db" in str(eng.url)
