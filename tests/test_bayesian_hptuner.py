import os
from types import SimpleNamespace
import types
from pathlib import Path

import pandas as pd
import pytest

from helpers import load_module_from_path


_HPTUNE_TRIAL_MODULE = load_module_from_path(
    "test_model_hptune_trial_for_bayes",
    Path(__file__).resolve().parents[1] / "src" / "model" / "hptune_trial.py",
)
_UTIL_HPTUNE_MODULE = load_module_from_path(
    "test_util_hptune_for_bayes",
    Path(__file__).resolve().parents[1] / "src" / "util" / "hptune.py",
)
_FAKE_UTIL_PACKAGE = types.ModuleType("util")
_FAKE_UTIL_PACKAGE.hptune = _UTIL_HPTUNE_MODULE
_MODULE = load_module_from_path(
    "test_model_bayesian_hptuner",
    Path(__file__).resolve().parents[1] / "src" / "model" / "bayesian_hptuner.py",
    injected_modules={
        "util": _FAKE_UTIL_PACKAGE,
        "util.hptune": _UTIL_HPTUNE_MODULE,
        "model.hptune_trial": _HPTUNE_TRIAL_MODULE,
    },
)
BayesianHPTuner = _MODULE.BayesianHPTuner
HPTuneTrial = _HPTUNE_TRIAL_MODULE.HPTuneTrial
TRIAL_LOG_COLUMNS = _UTIL_HPTUNE_MODULE.TRIAL_LOG_COLUMNS


def fake_settings(project_root: Path) -> SimpleNamespace:
    """Build a small settings object for isolated HPTune tests."""
    hptune = SimpleNamespace(
        dir=None,
        num_initial_trials=2,
        allowed_epochs=[10, 20, 30],
        allowed_batch_sizes=[16, 32, 64],
    )
    return SimpleNamespace(
        project_root=str(project_root),
        cfg=SimpleNamespace(hptune=hptune),
        default_hptune_param_bounds=lambda allowed_epochs, batch_sizes: {
            "lr": (1e-5, 1e-2),
            "dropout": (0.1, 0.5),
            "log_wd": (-6.0, -2.0),
            "epochs": (10.0, 30.0),
            "gradient_clip": (0.0, 5.0),
            "lr_scheduler_u": (0.0, 1.0),
            "lr_scheduler_factor": (0.1, 0.9),
            "lr_sched_patience": (1.0, 6.0),
            "early_stop_patience": (2.0, 10.0),
            "batch_idx": (0.0, 2.0),
        },
    )


def make_trial(**overrides) -> HPTuneTrial:
    """Create a reusable HPTune trial fixture with optional field overrides."""
    data = {
        "lr": 1e-3,
        "epochs": 20,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "gradient_clip": 1.0,
        "lr_scheduler": True,
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 2,
        "early_stopping_patience": 4,
        "trial_id": "trial_1",
        "val_loss": 0.25,
        "status": 0,
    }
    data.update(overrides)
    return HPTuneTrial(**data)


def make_tuner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BayesianHPTuner:
    """Construct a tuner rooted at a temporary project directory."""
    monkeypatch.delenv("DLDL_HPTUNE_DIR", raising=False)
    monkeypatch.setattr(_MODULE, "load_settings", lambda: fake_settings(tmp_path))
    return BayesianHPTuner()


def test_suggestion_to_trial_clamps_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clamp Bayesian suggestions into valid discrete and bounded trial values."""
    tuner = make_tuner(Path("/tmp"), monkeypatch)

    trial = tuner._suggestion_to_trial(
        {
            "lr": 2e-3,
            "epochs": 24.9,
            "dropout": 0.3,
            "log_wd": -5.0,
            "batch_idx": 1.6,
            "gradient_clip": 1.2,
            "lr_scheduler_u": 0.8,
            "lr_scheduler_factor": 1.5,
            "lr_sched_patience": 8.9,
            "early_stop_patience": 1.2,
        }
    )

    assert trial.epochs == 20
    assert trial.batch_size == 64
    assert trial.lr_scheduler is True
    assert trial.lr_scheduler_factor == pytest.approx(0.9)
    assert trial.lr_scheduler_patience == 6
    assert trial.early_stopping_patience == 2


def test_find_pending_trial_returns_first_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return the first running or queued trial row from the trials log."""
    tuner = make_tuner(tmp_path, monkeypatch)
    base_row = {
        "lr": 1e-3,
        "epochs": 10,
        "dropout": 0.1,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "gradient_clip": 1.0,
        "lr_scheduler": 1,
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 2,
        "early_stopping_patience": 4,
    }
    df = pd.DataFrame(
        [
            {**base_row, "trial_id": "trial_1", "val_loss": 0.5, "status": 0},
            {**base_row, "trial_id": "trial_2", "val_loss": -1.0, "status": -1},
            {**base_row, "trial_id": "trial_3", "val_loss": -1.0, "status": -2},
        ]
    )

    pending = tuner.find_pending_trial(df)

    assert pending is not None
    assert pending.trial_id == "trial_2"


def test_post_run_checkpoint_cleanup_block_mentions_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit shell cleanup that preserves only the best checkpoint artifact."""
    tuner = make_tuner(Path("/tmp"), monkeypatch)

    block = tuner._post_run_checkpoint_cleanup_block()

    assert '${JOB_ID}_best_params.pt' in block
    assert 'rm -f "$checkpoint"' in block


def test_sample_hyperparameters_uses_random_before_warmup_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use random sampling until the configured number of completed warmup trials is reached."""
    tuner = make_tuner(tmp_path, monkeypatch)
    random_trial = make_trial(trial_id=None, status=-1)
    monkeypatch.setattr(tuner, "sample_random", lambda: random_trial)
    monkeypatch.setattr(
        tuner,
        "sample_bayesian",
        lambda _df: pytest.fail("Bayesian sampling should not run during warmup"),
    )

    chosen = tuner.sample_hyperparameters(
        pd.DataFrame([make_trial(status=0).to_csv_row()])
    )

    assert chosen is random_trial


def test_update_trials_marks_completed_rows_and_syncs_best_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Promote completed running trials and trigger best-trial artifact syncing."""
    tuner = make_tuner(tmp_path, monkeypatch)
    sync_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        _MODULE,
        "parse_val_loss",
        lambda _trial_dir: (True, 0.125),
    )
    monkeypatch.setattr(
        _MODULE,
        "sync_best_trial_artifacts",
        lambda df, trials_dir, best_trial_dir: sync_calls.append(
            (trials_dir, best_trial_dir)
        ),
    )
    Path(tuner.trials_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([make_trial(status=-1, val_loss=-1.0).to_csv_row()])

    updated = tuner.update_trials(df.copy())

    assert updated.loc[0, "status"] == 0
    assert updated.loc[0, "val_loss"] == pytest.approx(0.125)
    assert sync_calls == [(tuner.trials_dir, tuner.best_trial_dir)]
    assert Path(tuner.csv_path).exists()


def test_create_trial_writes_env_and_run_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Materialize a trial directory with both a rendered env file and chained run script."""
    tuner = make_tuner(tmp_path, monkeypatch)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "run_train.sh").write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "#PBS -N dldl_train",
                "#PBS -o old.out",
                "#PBS -e old.err",
                "set -e",
                "# __HPTUNE_ENV_INJECT__",
                "# __HPTUNE_CD_OVERRIDE__",
                'cd "${PBS_O_WORKDIR:-$(pwd)}"',
            ]
        ),
        encoding="utf-8",
    )
    trial = make_trial(trial_id="trial_4", status=-1, val_loss=-1.0)

    created = tuner.create_trial(trial, ["BASE_VAR=1"])

    env_path = Path(tuner.trials_dir) / "trial_4" / ".env"
    run_path = Path(tuner.trials_dir) / "trial_4" / "run.sh"
    assert created == "trial_4"
    assert "BASE_VAR=1" in env_path.read_text(encoding="utf-8")
    assert "JOB_ID=trial_4" in env_path.read_text(encoding="utf-8")
    run_text = run_path.read_text(encoding="utf-8")
    assert "source " in run_text
    assert 'BEST_PARAMS_PATH="$PROG_DIR/${JOB_ID}_best_params.pt"' in run_text
    assert "Submitting next controller" in run_text
    assert os.access(run_path, os.X_OK)


def test_run_reuses_pending_trial_without_creating_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-emit the pending trial marker instead of scheduling a new trial."""
    tuner = make_tuner(tmp_path, monkeypatch)
    pending_trial = make_trial(trial_id="trial_2", status=-1, val_loss=-1.0)
    df = pd.DataFrame([pending_trial.to_csv_row()])
    logged: list[str] = []

    monkeypatch.setattr(_MODULE, "load_trials", lambda *_args: df)
    monkeypatch.setattr(tuner, "update_trials", lambda in_df: in_df)
    monkeypatch.setattr(tuner, "find_pending_trial", lambda _df: pending_trial)
    monkeypatch.setattr(
        tuner,
        "_log_pass_hyperparameters",
        lambda trial, context: logged.append(f"{context}:{trial.trial_id}"),
    )
    monkeypatch.setattr(
        tuner,
        "_log_next_trial_marker",
        lambda dir_name: logged.append(f"marker:{dir_name}"),
    )
    monkeypatch.setattr(
        tuner,
        "create_trial",
        lambda *_args, **_kwargs: pytest.fail("create_trial should not run"),
    )

    tuner.run()

    assert logged == [
        "pending (resume / awaiting worker):trial_2",
        "marker:trial_2",
    ]


def test_run_stops_when_max_trials_are_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Avoid scheduling a new trial once the maximum row count has been reached."""
    monkeypatch.setenv("HPTUNE_MAX_TRIALS", "1")
    tuner = make_tuner(tmp_path, monkeypatch)
    df = pd.DataFrame([make_trial().to_csv_row()])

    monkeypatch.setattr(_MODULE, "load_trials", lambda *_args: df)
    monkeypatch.setattr(tuner, "update_trials", lambda in_df: in_df)
    monkeypatch.setattr(tuner, "find_pending_trial", lambda _df: None)
    monkeypatch.setattr(
        tuner,
        "sample_hyperparameters",
        lambda _df: pytest.fail("No new sampling should occur at max trials"),
    )

    tuner.run()


def test_run_appends_a_new_trial_row_to_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Append a newly proposed trial row after successfully materializing trial files."""
    tuner = make_tuner(tmp_path, monkeypatch)
    Path(tuner.trials_dir).mkdir(parents=True, exist_ok=True)
    empty_df = pd.DataFrame(columns=TRIAL_LOG_COLUMNS)
    created: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(_MODULE, "load_trials", lambda *_args: empty_df)
    monkeypatch.setattr(tuner, "update_trials", lambda in_df: in_df)
    monkeypatch.setattr(tuner, "find_pending_trial", lambda _df: None)
    monkeypatch.setattr(
        _MODULE, "next_trial_numbered_id", lambda *_args: "trial_9"
    )
    monkeypatch.setattr(
        tuner,
        "sample_hyperparameters",
        lambda _df: make_trial(trial_id=None, status=-1, val_loss=-1.0),
    )
    monkeypatch.setattr(_MODULE, "load_env_template", lambda _root: ["BASE=1"])
    monkeypatch.setattr(
        tuner,
        "create_trial",
        lambda trial, env_lines: created.append((trial.trial_id, env_lines)) or trial.trial_id,
    )

    tuner.run()

    written = pd.read_csv(tuner.csv_path)
    assert written["trial_id"].tolist() == ["trial_9"]
    assert int(written.loc[0, "status"]) == -1
    assert created == [("trial_9", ["BASE=1"])]
