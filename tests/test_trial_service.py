"""Tests for :class:`service.trial_service.TrialService` CRUD API."""

import os
import tempfile

import pytest

from database import default_trial_db_path
from model.hptune_trial import HPTuneTrial
from service.trial_service import TrialService


def _sample_trial(trial_id: str = "trial_1", **kwargs) -> HPTuneTrial:
    base = dict(
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
        trial_id=trial_id,
        val_loss=0.5,
        status=0,
        retries=0,
    )
    base.update(kwargs)
    return HPTuneTrial(**base)


def test_default_path_matches_database_module():
    with tempfile.TemporaryDirectory() as tmp:
        assert TrialService.default_path(tmp) == default_trial_db_path(tmp)


def test_get_trials_empty():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        svc = TrialService(db_path)
        assert svc.get_trials() == []


def test_create_get_trials_get_trial():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        svc = TrialService(db_path)
        t = _sample_trial()
        svc.create_trial(t)
        all_rows = svc.get_trials()
        assert len(all_rows) == 1
        assert all_rows[0].trial_id == "trial_1"
        one = svc.get_trial("trial_1")
        assert one is not None
        assert one.val_loss == pytest.approx(0.5)


def test_create_duplicate_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        svc = TrialService(db_path)
        svc.create_trial(_sample_trial())
        with pytest.raises(ValueError, match="already exists"):
            svc.create_trial(_sample_trial())


def test_update_trial():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        svc = TrialService(db_path)
        svc.create_trial(_sample_trial(val_loss=1.0))
        updated = svc.update_trial(
            "trial_1",
            _sample_trial(trial_id="ignored", val_loss=0.25, status=1),
        )
        assert updated.trial_id == "trial_1"
        assert updated.val_loss == pytest.approx(0.25)
        row = svc.get_trial("trial_1")
        assert row is not None
        assert row.val_loss == pytest.approx(0.25)
        assert row.status == 1


def test_update_trial_missing_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        svc = TrialService(db_path)
        with pytest.raises(KeyError):
            svc.update_trial("missing", _sample_trial(trial_id="missing"))


def test_delete_trial():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        svc = TrialService(db_path)
        svc.create_trial(_sample_trial())
        assert svc.delete_trial("trial_1") is True
        assert svc.get_trial("trial_1") is None
        assert svc.delete_trial("trial_1") is False


def test_persist_snapshot_replaces_table():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        svc = TrialService(db_path)
        svc.create_trial(_sample_trial("a"))
        svc.create_trial(_sample_trial("b"))
        svc.persist_snapshot([_sample_trial("only")])
        ids = [t.trial_id for t in svc.get_trials()]
        assert ids == ["only"]
