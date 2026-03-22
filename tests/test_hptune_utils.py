import os
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from model import HPTuneTrial


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "util" / "hptune.py"
_SPEC = importlib.util.spec_from_file_location("test_util_hptune", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HPTUNE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HPTUNE)

TRIAL_LOG_COLUMNS = _HPTUNE.TRIAL_LOG_COLUMNS
best_checkpoint_path = _HPTUNE.best_checkpoint_path
create_run_script = _HPTUNE.create_run_script
load_env_template = _HPTUNE.load_env_template
load_trials = _HPTUNE.load_trials
next_trial_numbered_id = _HPTUNE.next_trial_numbered_id
parse_val_loss = _HPTUNE.parse_val_loss
sync_best_trial_artifacts = _HPTUNE.sync_best_trial_artifacts


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
        "trial_id": "trial_1",
        "val_loss": 0.123,
        "status": 0,
    }
    data.update(overrides)
    return HPTuneTrial(**data)


def test_load_env_template_filters_comments_and_overrides(tmp_path: Path) -> None:
    """Keep shared env lines while filtering comments and per-trial override keys."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "DATA_DIR=data/raw",
                "LEARNING_RATE=0.1",
                "JOB_ID=trial_1",
                "CPU_USE=0.5",
            ]
        )
    )

    lines = load_env_template(str(tmp_path))

    assert lines == ["DATA_DIR=data/raw", "CPU_USE=0.5"]


def test_create_run_script_replaces_sentinels(tmp_path: Path) -> None:
    """Replace PBS/env sentinels when building a trial-specific run script."""
    template = tmp_path / "run_train.sh"
    template.write_text(
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
        )
    )

    script = create_run_script(
        project_root="/project/root",
        trial_dir="/project/root/data/hptune/trials/trial_1",
        env_path="/project/root/data/hptune/trials/trial_1/.env",
        template_path=str(template),
    )

    assert "# __HPTUNE_ENV_INJECT__" not in script
    assert "# __HPTUNE_CD_OVERRIDE__" not in script
    assert "#PBS -N dldl_hptune" in script
    assert "#PBS -o /dev/null" in script
    assert "#PBS -e /dev/null" in script
    assert "source /project/root/data/hptune/trials/trial_1/.env" in script
    assert 'exec > >(tee "$PROG_DIR/train_${PBS_JOBID}.log") 2>&1' in script
    assert "cd /project/root" in script


def test_next_trial_numbered_id_uses_max_of_csv_and_filesystem(tmp_path: Path) -> None:
    """Advance past the highest trial number seen in CSV rows or trial directories."""
    (tmp_path / "trial_4").mkdir()
    df = pd.DataFrame([{"trial_id": "trial_2"}])

    with pytest.warns(UserWarning, match="Filesystem has trial_4"):
        trial_id = next_trial_numbered_id(str(tmp_path), df)

    assert trial_id == "trial_5"


def test_load_trials_creates_new_log_with_expected_columns(tmp_path: Path) -> None:
    """Initialize an empty trials log with the expected CSV schema."""
    trials_dir = tmp_path / "trials"
    csv_path = trials_dir / "trials_log.csv"

    df = load_trials(str(trials_dir), str(csv_path))

    assert list(df.columns) == TRIAL_LOG_COLUMNS
    assert csv_path.exists()


def test_parse_val_loss_prefers_newest_log(tmp_path: Path) -> None:
    """Parse validation loss from the newest matching training log CSV."""
    older = tmp_path / "a_training_log.csv"
    newer = tmp_path / "b_training_log.csv"
    pd.DataFrame({"validation_loss": [0.8, 0.6]}).to_csv(older, index=False)
    pd.DataFrame({"validation_loss": [0.5, 0.2]}).to_csv(newer, index=False)

    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (1_000_100, 1_000_100))

    completed, val_loss = parse_val_loss(str(tmp_path))

    assert completed is True
    assert val_loss == pytest.approx(0.2)


def test_best_checkpoint_path_returns_existing_checkpoint(tmp_path: Path) -> None:
    """Return the best-checkpoint path when the expected artifact exists."""
    trial = make_trial(trial_id="trial_3")
    checkpoint = tmp_path / "trial_3" / "trial_3_best_params.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_text("weights")

    path = best_checkpoint_path(str(tmp_path), trial)

    assert path == str(checkpoint)


def test_sync_best_trial_artifacts_replaces_existing_snapshot(tmp_path: Path) -> None:
    """Refresh the best-trial snapshot while preserving unrelated files."""
    trials_dir = tmp_path / "trials"
    best_trial_dir = tmp_path / "best_trial"
    trial_dir = trials_dir / "trial_1"
    trial_dir.mkdir(parents=True)
    (trial_dir / ".env").write_text("JOB_ID=trial_1\n")
    checkpoint = trial_dir / "trial_1_best_params.pt"
    checkpoint.write_text("best-weights")

    best_trial_dir.mkdir()
    (best_trial_dir / ".env").write_text("stale\n")
    (best_trial_dir / "old_best_params.pt").write_text("stale-weights")
    (best_trial_dir / "keep.txt").write_text("keep-me")

    df = pd.DataFrame([make_trial().to_csv_row()])
    sync_best_trial_artifacts(df, str(trials_dir), str(best_trial_dir))

    assert (best_trial_dir / ".env").read_text() == "JOB_ID=trial_1\n"
    assert (best_trial_dir / "trial_1_best_params.pt").read_text() == "best-weights"
    assert not (best_trial_dir / "old_best_params.pt").exists()
    assert (best_trial_dir / "keep.txt").read_text() == "keep-me"
