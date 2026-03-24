import os
from dataclasses import asdict
from types import SimpleNamespace
import types
from pathlib import Path

import pytest

from helpers import load_module_from_path
from service.trial_service import TrialService


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
SerialTrial = _HPTUNE_TRIAL_MODULE.SerialTrial
ParallelTrial = _HPTUNE_TRIAL_MODULE.ParallelTrial


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
        "retries": 0,
    }
    data.update(overrides)
    return HPTuneTrial(**data)


def make_serial_trial(**overrides) -> SerialTrial:
    """Same as :func:`make_trial` but as :class:`SerialTrial` for materialization tests."""
    return SerialTrial(**asdict(make_trial(**overrides)))


def make_parallel_trial(**overrides) -> ParallelTrial:
    """Same as :func:`make_trial` but as :class:`ParallelTrial` for materialization tests."""
    return ParallelTrial(**asdict(make_trial(**overrides)))


def make_tuner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BayesianHPTuner:
    """Construct a tuner rooted at a temporary project directory."""
    monkeypatch.delenv("DLDL_HPTUNE_DIR", raising=False)
    monkeypatch.delenv("HPTUNE_CONTROLLER_NODES", raising=False)
    monkeypatch.delenv("HPTUNE_TRIAL_NODES", raising=False)
    monkeypatch.delenv("HPTUNE_RANDOM_INSERT_EVERY", raising=False)
    monkeypatch.delenv("HPTUNE_EI_XI", raising=False)
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
    assert isinstance(trial, SerialTrial)


def test_suggestion_to_trial_uses_parallel_class_when_parallelism_gt_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-slot parallelism builds :class:`ParallelTrial` proposals."""
    monkeypatch.setenv("HPTUNE_CONTROLLER_NODES", "5")
    monkeypatch.setenv("HPTUNE_TRIAL_NODES", "2")
    monkeypatch.delenv("DLDL_HPTUNE_DIR", raising=False)
    monkeypatch.delenv("HPTUNE_RANDOM_INSERT_EVERY", raising=False)
    monkeypatch.delenv("HPTUNE_EI_XI", raising=False)
    monkeypatch.setattr(_MODULE, "load_settings", lambda: fake_settings(tmp_path))

    tuner = BayesianHPTuner()
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
    assert isinstance(trial, ParallelTrial)


def test_tuner_reads_expected_improvement_xi_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the environment override for Expected Improvement xi."""
    monkeypatch.setenv("HPTUNE_EI_XI", "0.2")
    monkeypatch.delenv("DLDL_HPTUNE_DIR", raising=False)
    monkeypatch.delenv("HPTUNE_RANDOM_INSERT_EVERY", raising=False)
    monkeypatch.setattr(_MODULE, "load_settings", lambda: fake_settings(tmp_path))

    tuner = BayesianHPTuner()

    assert tuner.expected_improvement_xi == pytest.approx(0.2)


def test_tuner_reads_parallelism_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Derive dispatcher parallelism from worker-node slots per trial."""
    monkeypatch.setenv("HPTUNE_CONTROLLER_NODES", "5")
    monkeypatch.setenv("HPTUNE_TRIAL_NODES", "2")
    monkeypatch.delenv("DLDL_HPTUNE_DIR", raising=False)
    monkeypatch.delenv("HPTUNE_RANDOM_INSERT_EVERY", raising=False)
    monkeypatch.delenv("HPTUNE_EI_XI", raising=False)
    monkeypatch.setattr(_MODULE, "load_settings", lambda: fake_settings(tmp_path))

    tuner = BayesianHPTuner()

    assert tuner.parallelism == 2
    assert tuner._trial_cls is ParallelTrial


def test_post_run_checkpoint_cleanup_block_mentions_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit shell cleanup that preserves only the best checkpoint artifact."""
    _ = make_tuner(Path("/tmp"), monkeypatch)
    block = HPTuneTrial._post_run_checkpoint_cleanup_block()

    assert "${JOB_ID}_best_params.pt" in block
    assert 'rm -f "$checkpoint"' in block


def test_sample_hyperparameters_uses_random_before_warmup_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use random sampling until the configured number of completed warmup trials is reached."""
    tuner = make_tuner(tmp_path, monkeypatch)
    random_trial = make_trial(trial_id=None, status=-1)
    monkeypatch.setattr(
        tuner,
        "_sample_unique_random",
        lambda seen_signatures, context: random_trial,
    )
    monkeypatch.setattr(
        tuner,
        "sample_bayesian",
        lambda _trials: pytest.fail("Bayesian sampling should not run during warmup"),
    )

    chosen = tuner.sample_hyperparameters(
        [make_trial(status=0)],
    )

    assert chosen is random_trial


def test_sample_hyperparameters_inserts_random_trial_periodically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a unique random trial at the configured post-warmup cadence."""
    tuner = make_tuner(tmp_path, monkeypatch)
    tuner.random_insert_every = 3
    random_trial = make_trial(
        trial_id=None,
        status=-1,
        val_loss=-1.0,
        lr=2e-3,
        epochs=30,
    )
    monkeypatch.setattr(
        tuner,
        "_sample_unique_random",
        lambda seen_signatures, context: random_trial,
    )
    monkeypatch.setattr(
        tuner,
        "sample_bayesian",
        lambda _trials: pytest.fail(
            "Bayesian sampling should be skipped for periodic exploration"
        ),
    )
    trials = [
        make_trial(
            trial_id=f"trial_{index}", status=0, epochs=10 + 10 * (index % 3)
        )
        for index in range(1, 6)
    ]

    chosen = tuner.sample_hyperparameters(trials)

    assert chosen is random_trial


def test_sample_bayesian_falls_back_to_random_for_duplicate_suggestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fallback to random sampling when BO proposes an already-seen trial."""
    tuner = make_tuner(tmp_path, monkeypatch)
    existing = make_trial(trial_id="trial_1", status=0, val_loss=0.25)
    fallback_trial = make_trial(
        trial_id=None,
        status=-1,
        val_loss=-1.0,
        lr=3e-3,
        epochs=30,
        batch_size=64,
    )
    trials = [existing]

    class FakeOptimizer:
        """Return a duplicate suggestion while accepting duplicate registration."""

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def register(self, params, target) -> None:
            return None

        def suggest(self) -> dict[str, float]:
            return existing.bayesian_params(tuner.batch_sizes)

    monkeypatch.setattr(_MODULE, "BayesianOptimization", FakeOptimizer)
    monkeypatch.setattr(
        tuner,
        "_sample_unique_random",
        lambda seen_signatures, context: fallback_trial,
    )

    chosen = tuner.sample_bayesian(trials)

    assert chosen is fallback_trial


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
        lambda trials, trials_dir, best_trial_dir: sync_calls.append(
            (trials_dir, best_trial_dir)
        ),
    )
    Path(tuner.trials_dir).mkdir(parents=True, exist_ok=True)
    trial = make_trial(trial_id="trial_1", status=-1, val_loss=-1.0)
    os.makedirs(trial.path_under(tuner.trials_dir), exist_ok=True)

    updated = tuner.update_trials([trial])

    assert updated[0].status == 0
    assert updated[0].val_loss == pytest.approx(0.125)
    assert sync_calls == [(tuner.trials_dir, tuner.best_trial_dir)]
    assert TrialService(tuner.trials_db_path).get_trials()[0].status == 0


def test_serial_trial_materialize_writes_env_and_run_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Materialize a serial trial with both a rendered env file and chained run script."""
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
    trial = make_serial_trial(trial_id="trial_4", status=-1, val_loss=-1.0)

    created = trial.materialize_trial_files(
        project_root=tuner.project_root,
        trials_dir=tuner.trials_dir,
        env_lines=["BASE_VAR=1"],
    )

    env_path = Path(tuner.trials_dir) / "trial_4" / ".env"
    run_path = Path(tuner.trials_dir) / "trial_4" / "run.sh"
    assert created == "trial_4"
    assert "BASE_VAR=1" in env_path.read_text(encoding="utf-8")
    assert "JOB_ID=trial_4" in env_path.read_text(encoding="utf-8")
    run_text = run_path.read_text(encoding="utf-8")
    assert "source " in run_text
    assert 'BEST_PARAMS_PATH="$PROG_DIR/${JOB_ID}_best_params.pt"' in run_text
    assert '"$PROJECT_ROOT/scripts/controller.sh"' in run_text
    assert "HPTUNE_SERIAL_CONTROLLER_PATH" not in run_text
    assert os.access(run_path, os.X_OK)


def test_parallel_trial_materialize_skips_serial_chain_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Materialize a parallel trial without appending the serial controller handoff block."""
    tuner = make_tuner(tmp_path, monkeypatch)
    tuner.parallelism = 4
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
    trial = make_parallel_trial(trial_id="trial_5", status=-1, val_loss=-1.0)

    created = trial.materialize_trial_files(
        project_root=tuner.project_root,
        trials_dir=tuner.trials_dir,
        env_lines=["BASE_VAR=1"],
    )

    run_path = Path(tuner.trials_dir) / "trial_5" / "run.sh"
    run_text = run_path.read_text(encoding="utf-8")
    assert created == "trial_5"
    assert "Submitting next controller" not in run_text
    assert "HPTUNE_SERIAL_CONTROLLER_PATH" not in run_text


def test_mark_trials_running_updates_queued_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Promote queued trials to running after worker submission succeeds."""
    tuner = make_tuner(tmp_path, monkeypatch)
    Path(tuner.trials_dir).mkdir(parents=True, exist_ok=True)
    svc = TrialService(tuner.trials_db_path)
    svc.persist_snapshot(
        [
            make_trial(trial_id="trial_1", status=-2, val_loss=-1.0),
            make_trial(trial_id="trial_2", status=0),
        ]
    )

    tuner.mark_trials_running(["trial_1"])

    trials = TrialService(tuner.trials_db_path).get_trials()
    t1 = next(t for t in trials if t.trial_id == "trial_1")
    assert t1.status == -1


def test_mark_trial_failed_requeues_when_retries_remain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requeue failed trials immediately until the retry budget is exhausted."""
    monkeypatch.setenv("HPTUNE_MAX_RETRIES", "2")
    tuner = make_tuner(tmp_path, monkeypatch)
    Path(tuner.trials_dir).mkdir(parents=True, exist_ok=True)
    TrialService(tuner.trials_db_path).persist_snapshot(
        [
            make_trial(
                trial_id="trial_1", status=-1, val_loss=-1.0, retries=0
            ),
        ]
    )

    tuner.mark_trial_failed("trial_1", return_code=3)

    trials = TrialService(tuner.trials_db_path).get_trials()
    row = next(t for t in trials if t.trial_id == "trial_1")
    assert row.status == -2
    assert row.retries == 1


def test_mark_trial_failed_marks_permanent_failure_after_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mark a trial permanently failed once it has exhausted retry attempts."""
    monkeypatch.setenv("HPTUNE_MAX_RETRIES", "1")
    tuner = make_tuner(tmp_path, monkeypatch)
    Path(tuner.trials_dir).mkdir(parents=True, exist_ok=True)
    TrialService(tuner.trials_db_path).persist_snapshot(
        [
            make_trial(
                trial_id="trial_1", status=-1, val_loss=-1.0, retries=1
            ),
        ]
    )

    tuner.mark_trial_failed("trial_1", return_code=7)

    trials = TrialService(tuner.trials_db_path).get_trials()
    row = next(t for t in trials if t.trial_id == "trial_1")
    assert row.status == -3
    assert row.retries == 1


def test_run_stops_when_max_trials_are_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Avoid scheduling a new trial once the maximum row count has been reached."""
    monkeypatch.setenv("HPTUNE_MAX_TRIALS", "1")
    tuner = make_tuner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        tuner._trial_service,
        "get_trials",
        lambda: [make_trial()],
    )
    monkeypatch.setattr(tuner, "update_trials", lambda trials: trials)
    monkeypatch.setattr(
        tuner,
        "sample_hyperparameters",
        lambda _trials: pytest.fail("No new sampling should occur at max trials"),
    )

    tuner.run()


def test_run_appends_a_new_trial_to_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persist a newly proposed trial after successfully materializing trial files."""
    tuner = make_tuner(tmp_path, monkeypatch)
    Path(tuner.trials_dir).mkdir(parents=True, exist_ok=True)
    created = []

    monkeypatch.setattr(
        tuner._trial_service,
        "get_trials",
        lambda: [],
    )
    monkeypatch.setattr(tuner, "update_trials", lambda trials: trials)
    monkeypatch.setattr(_MODULE, "next_trial_numbered_id", lambda *_args: "trial_9")
    monkeypatch.setattr(
        tuner,
        "sample_hyperparameters",
        lambda _trials: make_trial(trial_id=None, status=-1, val_loss=-1.0),
    )
    monkeypatch.setattr(_MODULE, "load_env_template", lambda _root: ["BASE=1"])

    def fake_serial(self, *, project_root, trials_dir, env_lines, log_dir=None):
        created.append((self.trial_id, env_lines))
        return self.trial_id

    monkeypatch.setattr(SerialTrial, "materialize_trial_files", fake_serial)

    tuner.run()

    written = TrialService(tuner.trials_db_path).get_trials()
    assert [t.trial_id for t in written] == ["trial_9"]
    assert written[0].status == -2
    assert created == [("trial_9", ["BASE=1"])]
