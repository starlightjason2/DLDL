import json
from pathlib import Path

import pytest

from config import settings as settings_module


def write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "hptune": {
                    "dir": "data/hptune",
                    "lr_min": 1e-5,
                    "lr_max": 1e-2,
                    "dropout_min": 0.1,
                    "dropout_max": 0.5,
                    "allowed_epochs": [10, 20, 30],
                    "num_initial_trials": 3,
                    "weight_decay_log_min": -6.0,
                    "weight_decay_log_max": -2.0,
                    "allowed_batch_sizes": [16, 32, 64],
                    "gradient_clip_min": 0.0,
                    "gradient_clip_max": 5.0,
                    "lr_scheduler_factor_min": 0.1,
                    "lr_scheduler_factor_max": 0.9,
                    "lr_scheduler_patience_min": 1,
                    "lr_scheduler_patience_max": 6,
                    "early_stopping_patience_min": 2,
                    "early_stopping_patience_max": 10,
                },
                "defaultTraining": {
                    "early_stopping_patience": 8,
                    "learning_rate": 0.001,
                    "num_epochs": 20,
                    "log_interval": 5,
                    "weight_decay": 1e-4,
                    "dropout_rate": 0.25,
                    "batch_size": 32,
                    "lr_scheduler": True,
                    "lr_scheduler_factor": 0.5,
                    "lr_scheduler_patience": 3,
                    "gradient_clip": 1.0,
                },
                "architecture": {
                    "conv1_filters": 16,
                    "conv1_kernel": 9,
                    "conv1_padding": 4,
                    "conv2_filters": 32,
                    "conv2_kernel": 5,
                    "conv2_padding": 2,
                    "conv3_filters": 64,
                    "conv3_kernel": 3,
                    "conv3_padding": 1,
                    "pool_size": 4,
                    "fc1_size": 120,
                    "fc2_size": 60,
                },
            }
        ),
        encoding="utf-8",
    )


def clear_settings_cache() -> None:
    settings_module._dotenv.cache_clear()
    settings_module.load_settings.cache_clear()


def test_load_settings_uses_custom_config_and_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load settings from a custom config file and merge env overrides into it."""
    config_path = tmp_path / "custom.json"
    write_config(config_path)
    clear_settings_cache()
    monkeypatch.setenv("DLDL_CONFIG", str(config_path))
    monkeypatch.setenv("LEARNING_RATE", "0.02")
    monkeypatch.setenv("FC1_SIZE", "256")

    settings = settings_module.load_settings()

    assert settings.dldl_config_path == str(config_path)
    assert settings.training_config.learning_rate == pytest.approx(0.02)
    assert settings.architecture_config.fc1_size == 256
    assert settings.project_root.endswith("DLDL")


def test_default_hptune_param_bounds_uses_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build default HPTune bounds from config values and explicit epoch/batch overrides."""
    config_path = tmp_path / "custom.json"
    write_config(config_path)
    clear_settings_cache()
    monkeypatch.setenv("DLDL_CONFIG", str(config_path))

    settings = settings_module.load_settings()
    bounds = settings.default_hptune_param_bounds(
        allowed_epochs=(5, 15),
        batch_sizes=(8, 16, 32, 64),
    )

    assert bounds["epochs"] == (5.0, 15.0)
    assert bounds["batch_idx"] == (0.0, 3.0)
    assert bounds["lr"] == (1e-5, 1e-2)


def test_load_settings_raises_for_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise a clear error when the requested DLDL config file does not exist."""
    clear_settings_cache()
    monkeypatch.setenv("DLDL_CONFIG", "/does/not/exist.json")

    with pytest.raises(FileNotFoundError, match="DLDL config missing"):
        settings_module.load_settings()
