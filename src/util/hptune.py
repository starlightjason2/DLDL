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
    """
    max_n = 0
    if "trial_id" in df.columns:
        for val in df["trial_id"].dropna():
            s = str(val).strip()
            m = _TRIAL_NUM_DIR_RE.match(s)
            if m:
                max_n = max(max_n, int(m.group(1)))
    if os.path.isdir(trials_dir):
        try:
            for name in os.listdir(trials_dir):
                path = os.path.join(trials_dir, name)
                if os.path.isdir(path):
                    m = _TRIAL_NUM_DIR_RE.match(name)
                    if m:
                        max_n = max(max_n, int(m.group(1)))
        except OSError:
            pass
    return f"trial_{max_n + 1}"


def parse_val_loss(trial_dir: str) -> tuple[bool, float]:
    """Parse best validation loss from training log CSV. Returns (completed, val_loss)."""
    for path in glob.glob(os.path.join(trial_dir, "*training_log.csv")):
        try:
            df = pd.read_csv(path)
            if not df.empty:
                best_idx = df["validation_loss"].idxmin()
                return True, float(df.loc[best_idx, "validation_loss"])
        except Exception:
            continue
    return False, -2.0


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


def load_env_template(project_root: str, skip_vars: tuple[str, ...] = ENV_SKIP_VARS) -> list[str]:
    """Load base .env lines, excluding vars we override per trial."""
    for name in (".env.polaris", ".env"):
        path = os.path.join(project_root, name)
        if os.path.exists(path):
            with open(path) as f:
                return [
                    line.rstrip()
                    for line in f
                    if line.strip()
                    and not line.strip().startswith("#")
                    and not any(line.strip().startswith(f"{v}=") for v in skip_vars)
                ]
    return []


def create_run_script(
    project_root: str,
    trial_dir: str,
    env_path: str,
    template_path: str,
) -> str:
    """Build run.sh content from template with trial-specific overrides."""
    with open(template_path) as f:
        script = f.read()
    script = (
        script.replace("#PBS -N dldl_train", "#PBS -N dldl_hptune")
        .replace('cd "${PBS_O_WORKDIR:-$(pwd)}"', f"cd {project_root}")
        .replace(
            "# .env is loaded by Python (config.settings.load_settings) when the script runs",
            f"set -a\nsource {env_path}\nset +a\n"
            f"# Single merged log via tee; discard PBS -o/-e to avoid duplicating the same stream\n"
            f"exec > >(tee \"$PROG_DIR/train_${{PBS_JOBID}}.log\") 2>&1",
        )
    )
    script = re.sub(r"#PBS -o .*", "#PBS -o /dev/null", script)
    script = re.sub(r"#PBS -e .*", "#PBS -e /dev/null", script)
    return script
