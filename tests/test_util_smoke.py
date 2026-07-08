"""Smoke tests for small pure helpers (filesystem/temp only; no training)."""

from __future__ import annotations

from pathlib import Path

import pytest

from model.hyperparam_space import hp_tune_mode
from util.best_model import resolve_best_model_checkpoint
from util.data_loading import env_int, env_tuple
from util.hp_tune import next_trial_numbered_id, write_env


def test_env_tuple_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HP_TUNE_ALLOWED_EPOCHS", "2, 4, 8")

    assert env_tuple("HP_TUNE_ALLOWED_EPOCHS") == (2, 4, 8)
    assert env_int("HP_TUNE_NUM_INITIAL_TRIALS") == 2


def test_hp_tune_mode() -> None:
    assert hp_tune_mode() == "training"


def test_hp_tune_mode_architecture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HP_TUNE_MODE", "architecture")

    assert hp_tune_mode() == "architecture"


def test_next_trial_numbered_id(tmp_path: Path) -> None:
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()
    (trials_dir / "trial_2").mkdir()

    assert next_trial_numbered_id(trials_dir, ["trial_1", "trial_2"]) == "trial_3"


def test_resolve_best_model_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "best_model"
    model_dir.mkdir()
    checkpoint = model_dir / "trial_1_best_params.pt"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setattr("util.best_model.best_model_dir", lambda: model_dir)

    resolved = resolve_best_model_checkpoint()

    assert resolved == checkpoint


def test_write_env(tmp_path: Path) -> None:
    env_path = tmp_path / "trial_1" / ".env"

    write_env(env_path, {"JOB_ID": "trial_1", "LEARNING_RATE": "0.001"})

    text = env_path.read_text(encoding="utf-8")
    assert "JOB_ID=" in text
    assert "LEARNING_RATE=" in text
