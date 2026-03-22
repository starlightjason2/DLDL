"""Utilities for DLDL Bayesian hyperparameter tuning."""

import glob
import os
import re

import pandas as pd

# Enumerated trial folders: trial_1, trial_2, ...
_TRIAL_NUM_DIR_RE = re.compile(r"^trial_(\d+)$")

# Vars set per trial in .env; must match create_trial in model/bayesian_hptuner.py
ENV_SKIP_VARS = (
    "LEARNING_RATE",
    "NUM_EPOCHS",
    "DROPOUT_RATE",
    "WEIGHT_DECAY",
    "BATCH_SIZE",
    "GRADIENT_CLIP",
    "LR_SCHEDULER",
    "LR_SCHEDULER_FACTOR",
    "LR_SCHEDULER_PATIENCE",
    "EARLY_STOPPING_PATIENCE",
    "PROG_DIR",
    "JOB_ID",
)

# trials_log.csv schema (trial_id = trial_N folder name; hyperparameters are columns, not the path)
TRIAL_LOG_COLUMNS = [
    "trial_id",
    "lr",
    "epochs",
    "dropout",
    "weight_decay",
    "batch_size",
    "gradient_clip",
    "lr_scheduler",
    "lr_scheduler_factor",
    "lr_scheduler_patience",
    "early_stopping_patience",
    "val_loss",
    "status",
]


def next_trial_numbered_id(trials_dir: str, df: pd.DataFrame) -> str:
    """Next sequential directory name: ``trial_1``, ``trial_2``, ...

    Uses the max index found in ``df['trial_id']`` and existing ``trials_dir/trial_*`` folders.
    Warns if the filesystem is ahead of the CSV (indicates a partial previous run).
    """
    csv_max = 0
    if "trial_id" in df.columns:
        for val in df["trial_id"].dropna():
            s = str(val).strip()
            m = _TRIAL_NUM_DIR_RE.match(s)
            if m:
                csv_max = max(csv_max, int(m.group(1)))

    fs_max = 0
    if os.path.isdir(trials_dir):
        try:
            for name in os.listdir(trials_dir):
                path = os.path.join(trials_dir, name)
                if os.path.isdir(path):
                    m = _TRIAL_NUM_DIR_RE.match(name)
                    if m:
                        fs_max = max(fs_max, int(m.group(1)))
        except OSError:
            pass

    if fs_max > csv_max:
        import warnings

        warnings.warn(
            f"Filesystem has trial_{fs_max} but CSV only knows trial_{csv_max}. "
            "This may indicate a partial previous run where the directory was created "
            "but the CSV row was never written. Skipping ahead to avoid collision.",
            stacklevel=2,
        )

    return f"trial_{max(csv_max, fs_max) + 1}"


def parse_val_loss(trial_dir: str) -> tuple[bool, float]:
    """Parse best validation loss from training log CSV. Returns (completed, val_loss).

    Sorts by mtime descending so the most recent log wins when multiple matches exist.
    Returns (False, nan) if no valid log is found.
    """
    candidates = sorted(
        glob.glob(os.path.join(trial_dir, "*training_log.csv")),
        key=os.path.getmtime,
        reverse=True,
    )
    for path in candidates:
        try:
            df = pd.read_csv(path)
            if not df.empty and "validation_loss" in df.columns:
                best_idx = df["validation_loss"].idxmin()
                return True, float(df.loc[best_idx, "validation_loss"])
        except Exception:
            continue
    return False, float("nan")


def load_trials(trials_dir: str, csv_path: str) -> pd.DataFrame:
    """Ensure trials log exists and return a dataframe with exactly ``TRIAL_LOG_COLUMNS``."""
    os.makedirs(trials_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=TRIAL_LOG_COLUMNS).to_csv(csv_path, index=False)
        return pd.read_csv(csv_path)
    df = pd.read_csv(csv_path)
    missing = [c for c in TRIAL_LOG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"trials_log.csv is missing columns {missing}; required schema: {TRIAL_LOG_COLUMNS}",
        )
    extra = [c for c in df.columns if c not in TRIAL_LOG_COLUMNS]
    if extra:
        df = df.drop(columns=extra)
    return df[TRIAL_LOG_COLUMNS]


def load_env_template(
    project_root: str, skip_vars: tuple[str, ...] = ENV_SKIP_VARS
) -> list[str]:
    """Load base `.env` lines, excluding per-trial override vars.

    HPTune uses the same `.env` entrypoint as the rest of the project. Per-trial vars in
    skip_vars are excluded since they are written directly into the trial's `.env` by
    create_trial.
    """
    lines: list[str] = []
    path = os.path.join(project_root, ".env")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required HPTune env template not found: {path}")

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or any(stripped.startswith(f"{v}=") for v in skip_vars)
            ):
                continue
            lines.append(line.rstrip())

    return lines


def create_run_script(
    project_root: str,
    trial_dir: str,
    env_path: str,
    template_path: str,
) -> str:
    """Build run.sh content from template with trial-specific overrides.

    Expects run_train.sh to contain two sentinel comments:
      # __HPTUNE_CD_OVERRIDE__   (line before the cd command)
      # __HPTUNE_ENV_INJECT__    (line before the .env comment, after set -e)

    If either sentinel is missing the substitution silently no-ops; the asserts below
    catch that at trial-creation time rather than at job-submission time.
    """
    with open(template_path) as f:
        script = f.read()

    inject_block = (
        f"set -a\nsource {env_path}\nset +a\n"
        f'exec > >(tee "$PROG_DIR/train_${{PBS_JOBID}}.log") 2>&1'
    )

    script = (
        script.replace("#PBS -N dldl_train", "#PBS -N dldl_hptune")
        .replace("# __HPTUNE_CD_OVERRIDE__", f"cd {project_root}")
        .replace("# __HPTUNE_ENV_INJECT__", inject_block)
    )
    script = re.sub(r"#PBS -o .*", "#PBS -o /dev/null", script)
    script = re.sub(r"#PBS -e .*", "#PBS -e /dev/null", script)

    assert "# __HPTUNE_CD_OVERRIDE__" not in script, (
        f"CD override sentinel was not replaced in {template_path}. "
        "Add '# __HPTUNE_CD_OVERRIDE__' on its own line before the cd command."
    )
    assert "# __HPTUNE_ENV_INJECT__" not in script, (
        f"Env inject sentinel was not replaced in {template_path}. "
        "Add '# __HPTUNE_ENV_INJECT__' on its own line after 'set -e'."
    )

    return script
