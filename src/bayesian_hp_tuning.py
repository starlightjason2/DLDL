#!/usr/bin/env python3
"""
Bayesian optimization for DLDL hyperparameter tuning on Polaris.

Workflow:
    1) Load or initialize trials_log.csv
    2) Update in-progress trials by parsing training log CSVs
    3) If completed trials < NUM_INITIAL_TRIALS, sample randomly; else use BO
    4) Create trial dir with run.sh and .env, print trial name for controller
"""

import argparse
import os

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

from constants import (
    ALLOWED_EPOCHS,
    DROPOUT_MIN,
    DROPOUT_MAX,
    HPTUNE_CSV_PATH,
    LR_MIN,
    LR_MAX,
    NUM_INITIAL_TRIALS,
    TRIALS_DIR,
    _PROJECT_ROOT,
)
from util.hptune import (
    create_run_script,
    load_env_template,
    load_trials,
    parse_val_loss,
    trial_dir_name,
)


def update_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Update in-progress trials (val_loss=-1) by checking training logs."""
    for idx in df[df["val_loss"] == -1].index:
        row = df.loc[idx]
        path = os.path.join(TRIALS_DIR, trial_dir_name(row["lr"], int(row["epochs"]), row["dropout"]))
        completed, val_loss = parse_val_loss(path)
        if completed:
            df.loc[idx, ["val_loss", "status"]] = val_loss, 0
    df.to_csv(HPTUNE_CSV_PATH, index=False)
    return df


def sample_random() -> tuple[float, int, float]:
    """Sample hyperparameters uniformly at random."""
    lr = 10 ** np.random.uniform(np.log10(LR_MIN), np.log10(LR_MAX))
    epochs = int(np.random.choice(ALLOWED_EPOCHS))
    dropout = float(np.random.uniform(DROPOUT_MIN, DROPOUT_MAX))
    return lr, epochs, dropout


def sample_bayesian(df: pd.DataFrame) -> tuple[float, int, float]:
    """Bayesian optimization to suggest next hyperparams."""
    completed = df[df["status"] == 0]
    if completed.empty:
        return sample_random()

    pbounds = {
        "lr": (LR_MIN, LR_MAX),
        "epochs": (float(min(ALLOWED_EPOCHS)), float(max(ALLOWED_EPOCHS))),
        "dropout": (DROPOUT_MIN, DROPOUT_MAX),
    }
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        acquisition_function=acquisition.ExpectedImprovement(xi=0.0),
        verbose=0,
        random_state=42,
    )
    for _, row in completed.iterrows():
        optimizer.register(
            params={"lr": row["lr"], "epochs": float(row["epochs"]), "dropout": row["dropout"]},
            target=-row["val_loss"],
        )

    suggestion = optimizer.suggest()
    epochs = min(ALLOWED_EPOCHS, key=lambda x: abs(x - suggestion["epochs"]))
    return suggestion["lr"], epochs, suggestion["dropout"]


def sample_hyperparameters(df: pd.DataFrame) -> tuple[float, int, float]:
    """Choose next hyperparams: random if < NUM_INITIAL_TRIALS, else BO."""
    completed_count = (df["status"] == 0).sum()
    return sample_random() if completed_count < NUM_INITIAL_TRIALS else sample_bayesian(df)


def create_trial(lr: float, epochs: int, dropout: float, env_lines: list[str]) -> str:
    """Create trial directory with run.sh and .env. Returns trial_dir_name."""
    name = trial_dir_name(lr, epochs, dropout)
    trial_dir = os.path.join(TRIALS_DIR, name)
    os.makedirs(trial_dir, exist_ok=True)

    env_path = os.path.join(trial_dir, ".env")
    env_content = "\n".join(env_lines) + f"""

# HPTune overrides
LEARNING_RATE={lr}
NUM_EPOCHS={epochs}
DROPOUT_RATE={dropout}
PROG_DIR={trial_dir}
JOB_ID=training
"""
    with open(env_path, "w") as f:
        f.write(env_content)

    template_path = os.path.join(_PROJECT_ROOT, "scripts", "run_train.sh")
    script = create_run_script(_PROJECT_ROOT, trial_dir, env_path, template_path)
    run_path = os.path.join(trial_dir, "run.sh")
    with open(run_path, "w") as f:
        f.write(script)
    os.chmod(run_path, 0o755)
    return name


def find_pending_trial(df: pd.DataFrame) -> str | None:
    """Return trial_dir_name of first in-progress (-1) or unstarted (-2) trial, else None."""
    for status in (-1, -2):
        candidates = df[df["val_loss"] == status]
        if not candidates.empty:
            row = candidates.iloc[0]
            return trial_dir_name(row["lr"], int(row["epochs"]), row["dropout"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="DLDL Bayesian hyperparameter optimization")
    parser.add_argument("--chain-id", required=True, help="Unique chain identifier")
    parser.parse_args()

    df = update_trials(load_trials(TRIALS_DIR, HPTUNE_CSV_PATH))

    pending = find_pending_trial(df)
    if pending:
        print(f"Next trial -> {pending}")
        return

    lr, epochs, dropout = sample_hyperparameters(df)
    name = trial_dir_name(lr, epochs, dropout)

    new_row = pd.DataFrame({
        "lr": [lr], "epochs": [epochs], "dropout": [dropout],
        "val_loss": [-1], "status": [-1],
    })
    pd.concat([df, new_row], ignore_index=True).to_csv(HPTUNE_CSV_PATH, index=False)

    create_trial(lr, epochs, dropout, load_env_template(_PROJECT_ROOT))
    print(f"Next trial -> {name}")


if __name__ == "__main__":
    main()
