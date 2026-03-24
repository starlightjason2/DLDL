"""Tests for Pydantic :class:`schemas.trial_schema.TrialSchema` vs ORM/domain."""

import os
import tempfile

from database.trial_model import TrialTable
from service.trial_service import TrialService
from model.hptune_trial import HPTuneTrial
from schemas.trial_schema import TrialSchema


def _sample_hpt(trial_id: str = "trial_1") -> HPTuneTrial:
    return HPTuneTrial(
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


def test_trial_schema_from_attributes_orm_row():
    """``from_attributes`` loads from SQLAlchemy ``TrialTable`` (see SQLAlchemy ORM quick start)."""
    row = TrialTable(
        trial_id="trial_1",
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
        val_loss=0.5,
        status=0,
        retries=0,
    )
    schema = TrialSchema.model_validate(row)
    assert schema.lr_scheduler is True
    assert schema.trial_id == "trial_1"


def test_orm_roundtrip_matches_domain():
    """ORM row ↔ Pydantic ↔ domain round-trip preserves semantics."""
    t = _sample_hpt()
    orm = TrialTable(
        trial_id=t.trial_id,
        lr=t.lr,
        epochs=t.epochs,
        dropout=t.dropout,
        weight_decay=t.weight_decay,
        batch_size=t.batch_size,
        gradient_clip=t.gradient_clip,
        lr_scheduler=t.lr_scheduler,
        lr_scheduler_factor=t.lr_scheduler_factor,
        lr_scheduler_patience=t.lr_scheduler_patience,
        early_stopping_patience=t.early_stopping_patience,
        val_loss=t.val_loss,
        status=t.status,
        retries=t.retries,
    )
    assert orm.lr_scheduler is True
    schema = TrialSchema.model_validate(orm)
    assert schema.lr_scheduler is True
    back = HPTuneTrial(**schema.model_dump())
    assert back == t
    orm2 = TrialTable(**TrialSchema.model_validate(orm).model_dump())
    assert orm2.trial_id == orm.trial_id
    assert orm2.lr_scheduler == orm.lr_scheduler


def test_trial_schema_validates_fields_like_domain_constructor():
    """Constructing ``TrialSchema`` with the same kwargs as ``HPTuneTrial`` yields matching dump."""
    t = _sample_hpt()
    schema = TrialSchema(
        lr=t.lr,
        epochs=t.epochs,
        dropout=t.dropout,
        weight_decay=t.weight_decay,
        batch_size=t.batch_size,
        gradient_clip=t.gradient_clip,
        lr_scheduler=t.lr_scheduler,
        lr_scheduler_factor=t.lr_scheduler_factor,
        lr_scheduler_patience=t.lr_scheduler_patience,
        early_stopping_patience=t.early_stopping_patience,
        trial_id=t.trial_id,
        val_loss=t.val_loss,
        status=t.status,
        retries=t.retries,
    )
    u = HPTuneTrial(**schema.model_dump())
    assert u == t
    assert HPTuneTrial(**TrialSchema.model_validate(schema).model_dump()) == t


def test_hptune_trial_schema_roundtrip():
    t = _sample_hpt()
    s = TrialSchema(
        lr=t.lr,
        epochs=t.epochs,
        dropout=t.dropout,
        weight_decay=t.weight_decay,
        batch_size=t.batch_size,
        gradient_clip=t.gradient_clip,
        lr_scheduler=t.lr_scheduler,
        lr_scheduler_factor=t.lr_scheduler_factor,
        lr_scheduler_patience=t.lr_scheduler_patience,
        early_stopping_patience=t.early_stopping_patience,
        trial_id=t.trial_id,
        val_loss=t.val_loss,
        status=t.status,
        retries=t.retries,
    )
    assert s.model_dump()["lr"] == t.lr
    assert s.lr_scheduler == t.lr_scheduler


def test_database_list_trials_uses_schema_bridge():
    """Full DB path still returns domain trials equal to the inserted row."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "trials_log.db")
        t = _sample_hpt()
        svc = TrialService(db_path)
        svc.persist_snapshot([t])
        rows = svc.get_trials()
        assert len(rows) == 1
        assert rows[0] == t
