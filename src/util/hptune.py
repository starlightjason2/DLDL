"""Utilities for DLDL Bayesian hyperparameter tuning."""

import glob
import os
import re
import shutil
from collections.abc import Iterable
from typing import Sequence
from textwrap import dedent

import pandas as pd
from loguru import logger

from schemas.trial_schema import HPTuneTrial

# Enumerated trial folders: trial_1, trial_2, ...
_TRIAL_NUM_DIR_RE = re.compile(r"^trial_(\d+)$")


def next_trial_numbered_id(
    trials_dir: str,
    known_trial_ids: Iterable[str],
) -> str:
    """Next sequential directory name: ``trial_1``, ``trial_2``, ...

    Uses the max index in ``known_trial_ids`` and existing ``trials_dir/trial_*`` folders.
    Warns if the filesystem is ahead of the log (partial previous run).
    """
    log_max = 0
    for val in known_trial_ids:
        s = str(val).strip()
        m = _TRIAL_NUM_DIR_RE.match(s)
        if m:
            log_max = max(log_max, int(m.group(1)))

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

    if fs_max > log_max:
        import warnings

        warnings.warn(
            f"Filesystem has trial_{fs_max} but trial log only knows trial_{log_max}. "
            "This may indicate a partial previous run where the directory was created "
            "but the trial row was never written. Skipping ahead to avoid collision.",
            stacklevel=2,
        )

    return f"trial_{max(log_max, fs_max) + 1}"


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


def sync_best_trial_artifacts(
    trials: Sequence[HPTuneTrial],
    best_trial_dir: str,
) -> None:
    """Copy the current overall best trial's .env and checkpoint into best_trial/."""
    os.makedirs(best_trial_dir, exist_ok=True)

    completed = [t for t in trials if t.status == 0]
    if not completed:
        return

    best_trial = min(completed, key=lambda t: t.val_loss)
    best_trial_path = best_trial.dir_path
    env_source = os.path.join(best_trial_path, ".env")

    checkpoint_source = os.path.join(
        best_trial.dir_path, f"{best_trial.trial_id}_best_params.pt"
    )

    if not os.path.exists(env_source):
        logger.warning(
            "Best-trial sync skipped: missing .env for {} at {}",
            best_trial.trial_id,
            env_source,
        )
        return

    # Keep best_trial/ as a single-current-best snapshot instead of an archive.
    for existing_name in os.listdir(best_trial_dir):
        existing_path = os.path.join(best_trial_dir, existing_name)
        if not os.path.isfile(existing_path):
            continue
        if existing_name == ".env" or existing_name.endswith("_best_params.pt"):
            os.remove(existing_path)

    shutil.copy2(env_source, os.path.join(best_trial_dir, ".env"))
    checkpoint_name = os.path.basename(checkpoint_source)
    shutil.copy2(
        checkpoint_source,
        os.path.join(best_trial_dir, checkpoint_name),
    )
    logger.info(
        "Best-trial snapshot updated: trial_id={} val_loss={:.6f} -> {}",
        best_trial.trial_id,
        float(best_trial.val_loss),
        best_trial_dir,
    )
