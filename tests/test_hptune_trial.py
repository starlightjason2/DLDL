from pathlib import Path

import pandas as pd
import pytest

from model import HPTuneTrial


def make_trial(**overrides) -> HPTuneTrial:
    data = {
        "lr": 1e-3,
        "epochs": 25,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "gradient_clip": 1.5,
        "lr_scheduler": True,
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 4,
        "early_stopping_patience": 8,
        "trial_id": "trial_7",
        "val_loss": 0.123,
        "status": 0,
    }
    data.update(overrides)
    return HPTuneTrial(**data)


def test_dir_name_requires_trial_id() -> None:
    """Raise when a trial directory name is requested before assigning an ID."""
    trial = make_trial(trial_id=None)

    with pytest.raises(ValueError, match="trial_id must be set"):
        _ = trial.dir_name


def test_from_series_round_trip_preserves_values() -> None:
    """Round-trip a trial through CSV-style series serialization without data loss."""
    trial = make_trial()

    row = pd.Series(trial.to_csv_row())
    restored = HPTuneTrial.from_series(row)

    assert restored == trial


def test_bayesian_params_uses_nearest_batch_index() -> None:
    """Encode Bayesian parameters with the nearest allowed batch-size index."""
    trial = make_trial(batch_size=50, weight_decay=1e-6, lr_scheduler=False)

    params = trial.bayesian_params((32, 64, 128))

    assert params["batch_idx"] == 1.0
    assert params["log_wd"] == pytest.approx(-6.0)
    assert params["lr_scheduler_u"] == 0.0


def test_write_env_file_writes_expected_overrides(tmp_path: Path) -> None:
    """Write a trial env file with base lines plus HPTune override variables."""
    env_path = tmp_path / "trial_7" / ".env"
    env_path.parent.mkdir()
    trial = make_trial()

    trial.write_env_file(str(env_path), ["BASE_VAR=1", "OTHER_VAR=two"])

    content = env_path.read_text()
    assert "BASE_VAR=1" in content
    assert "LEARNING_RATE=0.001" in content
    assert "NUM_EPOCHS=25" in content
    assert "JOB_ID=trial_7" in content
    assert f"PROG_DIR={env_path.parent}" in content
    assert "TRAIN_LOGURU_FILE=0" in content
