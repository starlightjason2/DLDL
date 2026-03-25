"""Utilities for DLDL Bayesian hyperparameter tuning."""

from __future__ import annotations

import os
import re
import shutil
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from schemas.trial_schema import HPTuneTrial, TrialStatus

# Enumerated trial folders: trial_1, trial_2, ...
_TRIAL_NUM_DIR_RE = re.compile(r"^trial_(\d+)$")


def _trial_index(name: str) -> int | None:
    """Return the integer index from a ``trial_N`` name, or ``None``."""
    m = _TRIAL_NUM_DIR_RE.match(str(name).strip())
    return int(m.group(1)) if m else None


def next_trial_numbered_id(
    trials_dir: str | Path,
    known_trial_ids: Iterable[str],
) -> str:
    """Next sequential directory name: ``trial_1``, ``trial_2``, ...

    Uses the max index across ``known_trial_ids`` and existing ``trials_dir/trial_*``
    subdirectories. Warns if the filesystem is ahead of the log (partial previous run).
    """
    log_max = max(
        (_trial_index(tid) for tid in known_trial_ids if _trial_index(tid) is not None),
        default=0,
    )

    root = Path(trials_dir)
    fs_max = 0
    if root.is_dir():
        with logger.catch(OSError, message="Could not list trials_dir", reraise=False):
            fs_max = max(
                (
                    _trial_index(p.name)
                    for p in root.iterdir()
                    if p.is_dir() and _trial_index(p.name)
                ),
                default=0,
            )

    if fs_max > log_max:
        warnings.warn(
            f"Filesystem has trial_{fs_max} but trial log only knows trial_{log_max}. "
            "This may indicate a partial previous run where the directory was created "
            "but the trial row was never written. Skipping ahead to avoid collision.",
            stacklevel=2,
        )

    return f"trial_{max(log_max, fs_max) + 1}"


def parse_val_loss(trial_dir: str | Path) -> tuple[bool, float]:
    """Parse best validation loss from the most recent training log CSV.

    Returns ``(True, val_loss)`` on success, ``(False, nan)`` if no valid log is found.
    """
    candidates = sorted(
        Path(trial_dir).glob("*training_log.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        with logger.catch(
            Exception, message=f"Skipping unreadable log {path}", reraise=False
        ):
            df = pd.read_csv(path)
            if not df.empty and "validation_loss" in df.columns:
                return True, float(
                    df.loc[df["validation_loss"].idxmin(), "validation_loss"]
                )

    return False, float("nan")


def sync_best_trial_artifacts(
    trials: Sequence[HPTuneTrial],
    best_trial_dir: str | Path,
) -> None:
    """Copy the current overall best trial's ``.env`` and checkpoint into ``best_trial/``."""
    dest = Path(best_trial_dir)
    dest.mkdir(parents=True, exist_ok=True)

    completed = [t for t in trials if t.status == TrialStatus.COMPLETED]
    if not completed:
        return

    best = min(completed, key=lambda t: t.val_loss)
    env_src = Path(best.dir_path) / ".env"

    if not env_src.exists():
        logger.warning(
            "Best-trial sync skipped: missing .env for {} at {}", best.trial_id, env_src
        )
        return

    # Keep best_trial/ as a single-current-best snapshot — remove stale artefacts first.
    for existing in dest.iterdir():
        if existing.is_file() and (
            existing.name == ".env" or existing.name.endswith("_best_params.pt")
        ):
            existing.unlink()

    shutil.copy2(env_src, dest / ".env")

    checkpoint_src = Path(best.dir_path) / f"{best.trial_id}_best_params.pt"
    shutil.copy2(checkpoint_src, dest / checkpoint_src.name)

    logger.info(
        "Best-trial snapshot updated: trial_id={} val_loss={:.6f} -> {}",
        best.trial_id,
        float(best.val_loss),
        dest,
    )


def write_env(
    env_path: str | Path,
    env_keys: dict[str, Any],
    env_lines: list[str] | None = None,
) -> None:
    """Write a shell-sourceable ``.env`` file merging ``env_keys`` into the current environment.

    Values containing spaces are quoted. Writes atomically via ``Path.write_text``.
    """
    path = Path(env_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    merged = {**dict(os.environ), **env_keys}  # env_keys win over inherited env

    def _format(val: str) -> str:
        return f'"{val}"' if " " in val else val

    lines = (
        list(env_lines or [])
        + ["# Merged Environment"]
        + [f"{k}={_format(str(v))}" for k, v in merged.items()]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
