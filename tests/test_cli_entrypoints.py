import types
from pathlib import Path

from config import load_settings
from helpers import load_module_from_path


def test_config_package_exports_load_settings() -> None:
    """Expose ``load_settings`` from the config package namespace."""
    assert callable(load_settings)


def test_hptune_serial_main_calls_run(monkeypatch) -> None:
    """Invoke ``BayesianHPTuner.run`` exactly once from the serial HPTune CLI."""
    calls = {"run": 0}

    class FakeTuner:
        def run(self) -> None:
            calls["run"] += 1

    fake_hptuner_module = types.ModuleType("model.bayesian_hptuner")
    fake_hptuner_module.BayesianHPTuner = FakeTuner
    hptune_serial = load_module_from_path(
        "test_hptune_serial_cli",
        Path(__file__).resolve().parents[1] / "src" / "hptune_serial.py",
        injected_modules={"model.bayesian_hptuner": fake_hptuner_module},
    )

    hptune_serial.main()

    assert calls["run"] == 1


def test_hptune_serial_main_marks_running_trials(monkeypatch) -> None:
    """Route ``--mark-running`` arguments to the tuner without running dispatch."""
    calls = {"run": 0, "mark_running": None}

    class FakeTuner:
        def run(self) -> None:
            calls["run"] += 1

        def mark_trials_running(self, trial_ids) -> None:
            calls["mark_running"] = list(trial_ids)

    fake_hptuner_module = types.ModuleType("model.bayesian_hptuner")
    fake_hptuner_module.BayesianHPTuner = FakeTuner
    hptune_serial = load_module_from_path(
        "test_hptune_serial_cli_mark_running",
        Path(__file__).resolve().parents[1] / "src" / "hptune_serial.py",
        injected_modules={"model.bayesian_hptuner": fake_hptuner_module},
    )

    hptune_serial.main(["--mark-running", "trial_1", "trial_2"])

    assert calls["run"] == 0
    assert calls["mark_running"] == ["trial_1", "trial_2"]
