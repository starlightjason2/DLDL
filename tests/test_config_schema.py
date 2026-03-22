import pytest

from config.schema import ArchitectureConfig, DatasetEnv, HptuneConfig, TrainingConfig


def test_hptune_config_rejects_invalid_bounds() -> None:
    """Reject HPTune configs whose minimum learning rate is not below the maximum."""
    with pytest.raises(ValueError, match="lr_min must be < lr_max"):
        HptuneConfig(
            dir=None,
            lr_min=0.01,
            lr_max=0.01,
            dropout_min=0.1,
            dropout_max=0.5,
            allowed_epochs=[50],
            num_initial_trials=1,
            weight_decay_log_min=-6.0,
            weight_decay_log_max=-2.0,
            allowed_batch_sizes=[32],
            gradient_clip_min=0.0,
            gradient_clip_max=1.0,
            lr_scheduler_factor_min=0.1,
            lr_scheduler_factor_max=0.9,
            lr_scheduler_patience_min=1,
            lr_scheduler_patience_max=2,
            early_stopping_patience_min=1,
            early_stopping_patience_max=2,
        )


def test_training_config_merge_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override training defaults with environment variables during config merge."""
    defaults = TrainingConfig(
        early_stopping_patience=10,
        learning_rate=0.001,
        num_epochs=20,
        log_interval=5,
        weight_decay=1e-4,
        dropout_rate=0.2,
        batch_size=32,
        lr_scheduler=True,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=4,
        gradient_clip=1.0,
    )
    monkeypatch.setenv("LEARNING_RATE", "0.02")
    monkeypatch.setenv("NUM_EPOCHS", "99")
    monkeypatch.setenv("LR_SCHEDULER", "false")

    merged = TrainingConfig.merge_env(defaults)

    assert merged.learning_rate == pytest.approx(0.02)
    assert merged.num_epochs == 99
    assert merged.lr_scheduler is False
    assert merged.batch_size == defaults.batch_size


def test_architecture_config_merge_env_overrides_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Override architecture defaults with environment variables during config merge."""
    defaults = ArchitectureConfig(
        conv1_filters=16,
        conv1_kernel=9,
        conv1_padding=4,
        conv2_filters=32,
        conv2_kernel=5,
        conv2_padding=2,
        conv3_filters=64,
        conv3_kernel=3,
        conv3_padding=1,
        pool_size=4,
        fc1_size=120,
        fc2_size=60,
    )
    monkeypatch.setenv("FC1_SIZE", "256")
    monkeypatch.setenv("POOL_SIZE", "8")

    merged = ArchitectureConfig.merge_env(defaults)

    assert merged.fc1_size == 256
    assert merged.pool_size == 8
    assert merged.conv1_filters == defaults.conv1_filters


def test_dataset_env_reads_defaults_and_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read dataset env values from defaults first and patched environment second."""
    defaults = DatasetEnv.from_os()
    assert defaults.normalization_type == "meanvar-whole"

    monkeypatch.setenv("NORMALIZATION_TYPE", "scale")
    monkeypatch.setenv("CPU_USE", "0.75")
    monkeypatch.setenv("PREPROCESSOR_MAX_WORKERS", "7")

    env = DatasetEnv.from_os()

    assert env.normalization_type == "scale"
    assert env.cpu_use == pytest.approx(0.75)
    assert env.preprocessor_max_workers == 7
