"""Utilities for DLDL Bayesian hyperparameter tuning."""

import glob
import os
import re

import pandas as pd

ENV_SKIP_VARS = ("LEARNING_RATE", "NUM_EPOCHS", "DROPOUT_RATE", "PROG_DIR", "JOB_ID")


def trial_dir_name(lr: float, epochs: int, dropout: float) -> str:
    return f"lr_{lr:.2e}_epochs_{epochs}_dropout_{dropout:.2f}"


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
    """Ensure trials log exists and return dataframe."""
    os.makedirs(trials_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["lr", "epochs", "dropout", "val_loss", "status"]).to_csv(
            csv_path, index=False
        )
    return pd.read_csv(csv_path)


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
            "# .env is loaded by Python (constants.py) when the script runs",
            f"set -a\nsource {env_path}\nset +a\n"
            f"# Write our own log (don't rely on PBS -o/-e)\nexec > >(tee \"$PROG_DIR/train_${{PBS_JOBID}}.log\") 2>&1",
        )
    )
    script = re.sub(r"#PBS -o .*", f"#PBS -o {trial_dir}/train_%j.out", script)
    script = re.sub(r"#PBS -e .*", f"#PBS -e {trial_dir}/train_%j.err", script)
    return script
