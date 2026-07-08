"""Utilities for DLDL Bayesian hyperparameter tuning."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from model.trial_status import TrialStatus
from util.objective import PRECISION_COL, RECALL_COL, best_row, trial_metrics

if TYPE_CHECKING:
    from model.hp_trial import HpTuneTrial

# Training hyperparameters snapshotted into architecture trials (from project ``.env``).
_FIXED_TRAINING_ENV_KEYS = (
    "LEARNING_RATE",
    "NUM_EPOCHS",
    "LOG_INTERVAL",
    "WEIGHT_DECAY",
    "DROPOUT_RATE",
    "BATCH_SIZE",
    "LR_SCHEDULER",
    "LR_SCHEDULER_FACTOR",
    "LR_SCHEDULER_PATIENCE",
    "EARLY_STOPPING_PATIENCE",
    "GRADIENT_CLIP",
    "DATALOADER_NUM_WORKERS",
    "CLS_POS_WEIGHT",
    "DECISION_THRESHOLD",
    "MIN_PRECISION",
    "FBETA",
)


def fixed_training_env_keys() -> dict[str, str]:
    """Training env vars held fixed during ``HP_TUNE_MODE=architecture`` runs."""
    missing = [key for key in _FIXED_TRAINING_ENV_KEYS if key not in os.environ]
    if missing:
        raise KeyError(
            "Missing training hyperparameters required for architecture HP tune: "
            + ", ".join(missing)
        )
    return {key: os.environ[key] for key in _FIXED_TRAINING_ENV_KEYS}


def fixed_training_trial_fields() -> dict[str, Any]:
    """Map fixed project training env vars to :class:`HpTuneTrial` field names."""
    env = fixed_training_env_keys()
    return {
        "lr": float(env["LEARNING_RATE"]),
        "epochs": int(env["NUM_EPOCHS"]),
        "dropout": float(env["DROPOUT_RATE"]),
        "weight_decay": float(env["WEIGHT_DECAY"]),
        "batch_size": int(env["BATCH_SIZE"]),
        "gradient_clip": float(env["GRADIENT_CLIP"]),
        "lr_scheduler": env["LR_SCHEDULER"].lower() in ("true", "1", "yes", "on"),
        "lr_scheduler_factor": float(env["LR_SCHEDULER_FACTOR"]),
        "lr_scheduler_patience": int(env["LR_SCHEDULER_PATIENCE"]),
        "early_stopping_patience": int(env["EARLY_STOPPING_PATIENCE"]),
        "cls_pos_weight": float(env["CLS_POS_WEIGHT"]),
    }

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


def parse_trial_metrics(trial_dir: str | Path) -> tuple[bool, dict[str, float]]:
    """Parse trial metrics from the best validation epoch per ``util.objective``.

    The best epoch maximizes F-beta among rows with precision at or above
    ``MIN_PRECISION``. Returns ``(True, {"score", "recall", "precision", "f1"})`` on
    success, ``(False, {})`` if no valid log is found.
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
            if (
                not df.empty
                and PRECISION_COL in df.columns
                and RECALL_COL in df.columns
            ):
                best = best_row(df.to_dict("records"))
                return True, trial_metrics(best)

    return False, {}


def sync_best_trial_artifacts(
    trials: Sequence[HpTuneTrial],
    best_trial_dir: Path,
) -> None:
    """Refresh ``best_trial/`` with the current overall best trial's ``.env`` and checkpoint.

    The ``.env`` is regenerated from the trial itself rather than copied, so the sync
    never depends on a per-trial ``.env`` file being present (it is absent for trials
    trained outside the planner, e.g. via ``run_train.sh``). A missing checkpoint no
    longer blocks the rest of the snapshot.
    """
    dest = Path(best_trial_dir)
    dest.mkdir(parents=True, exist_ok=True)

    completed = [t for t in trials if t.status == TrialStatus.COMPLETED]
    if not completed:
        return

    best = max(completed, key=lambda t: t.score)

    # Regenerate the snapshot .env from the trial's own hyperparameters.
    write_env(str(dest / ".env"), best.trial_env_keys())

    checkpoint_src = Path(best.dir_path) / f"{best.trial_id}_best_params.pt"
    if not checkpoint_src.exists():
        logger.warning(
            "Best-trial sync: wrote .env but checkpoint missing for {} at {}",
            best.trial_id,
            checkpoint_src,
        )
        return

    # Replace any stale checkpoint so best_trial/ holds only the current best's.
    for existing in dest.glob("*_best_params.pt"):
        existing.unlink()
    shutil.copy2(checkpoint_src, dest / checkpoint_src.name)

    logger.info(
        "Best-trial snapshot updated: trial_id={} score={:.6f} -> {}",
        best.trial_id,
        float(best.score),
        dest,
    )


def write_env(
    env_path: str | Path,
    env_keys: dict[str, Any],
    env_lines: list[str] | None = None,
) -> None:
    """Write a shell-sourceable ``.env`` of explicit ``env_keys`` only (no process environ dump).

    Used for per-trial overrides; ``run.sh`` must ``source`` the project ``.env`` first for base vars.
    Values are ``shlex.quote``'d. Writes atomically via ``Path.write_text``.
    """
    path = Path(env_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _shell_value(val: str) -> str:
        return shlex.quote(str(val))

    lines = (
        list(env_lines or [])
        + ["# HP-tune trial overrides (see ``HpTuneTrial.trial_env_keys``)"]
        + [f"{k}={_shell_value(v)}" for k, v in env_keys.items()]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
