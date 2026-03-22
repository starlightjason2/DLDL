import types
from pathlib import Path

from config import load_settings
from helpers import load_module_from_path


def test_config_package_exports_load_settings() -> None:
    """Expose ``load_settings`` from the config package namespace."""
    assert callable(load_settings)


def test_bayesian_hp_tuning_main_calls_run(monkeypatch) -> None:
    """Invoke ``BayesianHPTuner.run`` exactly once from the CLI main entrypoint."""
    calls = {"run": 0}

    class FakeTuner:
        def run(self) -> None:
            calls["run"] += 1

    fake_hptuner_module = types.ModuleType("model.bayesian_hptuner")
    fake_hptuner_module.BayesianHPTuner = FakeTuner
    bayesian_hp_tuning = load_module_from_path(
        "test_bayesian_hp_tuning_cli",
        Path(__file__).resolve().parents[1] / "src" / "bayesian_hp_tuning.py",
        injected_modules={"model.bayesian_hptuner": fake_hptuner_module},
    )

    bayesian_hp_tuning.main()

    assert calls["run"] == 1
