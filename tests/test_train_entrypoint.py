from pathlib import Path
import runpy
import types

import pytest

from helpers import temporary_modules


class FakeLogger:
    """Minimal loguru-style logger used to execute training entrypoint modules in tests."""

    def remove(self) -> None:
        pass

    def add(self, *_args, **_kwargs) -> None:
        pass


def test_train_entrypoint_builds_model_and_dispatches_train_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Construct ``IpCNN`` from env/config values and pass rank/world-size to training."""
    prog_dir = tmp_path / "prog"
    data_path = tmp_path / "processed" / "dataset.pt"
    labels_path = tmp_path / "processed" / "labels.pt"

    monkeypatch.setenv("PROG_DIR", str(prog_dir))
    monkeypatch.setenv("DATA_PATH", str(data_path))
    monkeypatch.setenv("TRAIN_LABELS_PATH", str(labels_path))
    monkeypatch.setenv("JOB_ID", "job-123")
    monkeypatch.setenv("PMI_RANK", "2")
    monkeypatch.setenv("PMI_SIZE", "4")

    calls = {"init": None, "train": None}

    class FakeIpCNN:
        def __init__(self, **kwargs) -> None:
            calls["init"] = kwargs

        def train_model(self, **kwargs) -> None:
            calls["train"] = kwargs

    fake_schema = types.ModuleType("config.schema")
    fake_schema.DatasetEnv = types.SimpleNamespace(
        from_os=lambda: types.SimpleNamespace(normalization_type="meanvar-single")
    )
    fake_settings = types.ModuleType("config.settings")
    fake_settings.load_settings = lambda: types.SimpleNamespace(
        architecture_config=types.SimpleNamespace(
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
        ),
        training_config=types.SimpleNamespace(dropout_rate=0.35),
    )
    fake_cnn = types.ModuleType("model.cnn")
    fake_cnn.IpCNN = FakeIpCNN
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *_args, **_kwargs: None
    fake_loguru = types.ModuleType("loguru")
    fake_loguru.logger = FakeLogger()

    with temporary_modules(
        {
            "config.schema": fake_schema,
            "config.settings": fake_settings,
            "model.cnn": fake_cnn,
            "dotenv": fake_dotenv,
            "loguru": fake_loguru,
        }
    ):
        runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "src" / "train.py"),
            run_name="__main__",
        )

    assert calls["init"]["data_path"] == str(data_path)
    assert calls["init"]["labels_path"] == str(labels_path)
    assert calls["init"]["prog_dir"] == str(prog_dir)
    assert calls["init"]["normalization_type"] == "meanvar-single"
    assert calls["init"]["conv1"] == (16, 9, 4)
    assert calls["train"] == {"rank": 2, "world_size": 4, "job_id": "job-123"}


def test_train_entrypoint_defaults_rank_and_world_size_when_pmi_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default rank and world size to single-process values when PMI vars are absent."""
    prog_dir = tmp_path / "prog"
    data_path = tmp_path / "processed" / "dataset.pt"
    labels_path = tmp_path / "processed" / "labels.pt"

    monkeypatch.setenv("PROG_DIR", str(prog_dir))
    monkeypatch.setenv("DATA_PATH", str(data_path))
    monkeypatch.setenv("TRAIN_LABELS_PATH", str(labels_path))
    monkeypatch.setenv("JOB_ID", "job-456")
    monkeypatch.delenv("PMI_RANK", raising=False)
    monkeypatch.delenv("PMI_SIZE", raising=False)

    calls = {"train": None}

    class FakeIpCNN:
        def __init__(self, **_kwargs) -> None:
            pass

        def train_model(self, **kwargs) -> None:
            calls["train"] = kwargs

    fake_schema = types.ModuleType("config.schema")
    fake_schema.DatasetEnv = types.SimpleNamespace(
        from_os=lambda: types.SimpleNamespace(normalization_type="scale")
    )
    fake_settings = types.ModuleType("config.settings")
    fake_settings.load_settings = lambda: types.SimpleNamespace(
        architecture_config=types.SimpleNamespace(
            conv1_filters=8,
            conv1_kernel=5,
            conv1_padding=2,
            conv2_filters=8,
            conv2_kernel=5,
            conv2_padding=2,
            conv3_filters=8,
            conv3_kernel=3,
            conv3_padding=1,
            pool_size=2,
            fc1_size=16,
            fc2_size=8,
        ),
        training_config=types.SimpleNamespace(dropout_rate=0.1),
    )
    fake_cnn = types.ModuleType("model.cnn")
    fake_cnn.IpCNN = FakeIpCNN
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *_args, **_kwargs: None
    fake_loguru = types.ModuleType("loguru")
    fake_loguru.logger = FakeLogger()

    with temporary_modules(
        {
            "config.schema": fake_schema,
            "config.settings": fake_settings,
            "model.cnn": fake_cnn,
            "dotenv": fake_dotenv,
            "loguru": fake_loguru,
        }
    ):
        runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "src" / "train.py"),
            run_name="__main__",
        )

    assert calls["train"] == {"rank": 0, "world_size": 1, "job_id": "job-456"}
