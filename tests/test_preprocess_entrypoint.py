from pathlib import Path
import runpy
import types

import pytest

from helpers import temporary_modules


class FakeLogger:
    """Minimal loguru-style logger used to execute entrypoint modules in tests."""

    def remove(self) -> None:
        pass

    def add(self, *_args, **_kwargs) -> None:
        pass

    def info(self, *_args, **_kwargs) -> None:
        pass


def test_preprocess_entrypoint_rebuilds_cached_outputs_and_checks_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete stale preprocess outputs and run the dataset integrity check entrypoint."""
    prog_dir = tmp_path / "prog"
    data_path = tmp_path / "processed" / "dataset.pt"
    labels_path = tmp_path / "processed" / "labels.pt"
    prog_dir.mkdir()
    data_path.parent.mkdir(parents=True)
    data_path.write_text("stale", encoding="utf-8")
    labels_path.write_text("stale", encoding="utf-8")

    monkeypatch.setenv("PROG_DIR", str(prog_dir))
    monkeypatch.setenv("DATA_PATH", str(data_path))
    monkeypatch.setenv("TRAIN_LABELS_PATH", str(labels_path))

    calls = {"normalization_type": None, "scale_labels": None}

    class FakeDataset:
        def __init__(self, normalization_type: str) -> None:
            calls["normalization_type"] = normalization_type

        def check_dataset(self, scale_labels: bool = False) -> None:
            calls["scale_labels"] = scale_labels

    fake_schema = types.ModuleType("config.schema")
    fake_schema.DatasetEnv = types.SimpleNamespace(
        from_os=lambda: types.SimpleNamespace(normalization_type="scale")
    )
    fake_model_dataset = types.ModuleType("model.dataset")
    fake_model_dataset.IpDataset = FakeDataset
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *_args, **_kwargs: None
    fake_loguru = types.ModuleType("loguru")
    fake_loguru.logger = FakeLogger()

    with temporary_modules(
        {
            "config.schema": fake_schema,
            "model.dataset": fake_model_dataset,
            "dotenv": fake_dotenv,
            "loguru": fake_loguru,
        }
    ):
        runpy.run_path(
            str(
                Path(__file__).resolve().parents[1]
                / "src"
                / "preprocess_data.py"
            ),
            run_name="__main__",
        )

    assert not data_path.exists()
    assert not labels_path.exists()
    assert calls["normalization_type"] == "scale"
    assert calls["scale_labels"] is True
