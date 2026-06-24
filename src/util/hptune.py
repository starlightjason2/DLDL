"""Utilities for DLDL Bayesian hyperparameter tuning."""

from __future__ import annotations

import re
import shlex
import shutil
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from model.hp_trial import HPTuneTrial

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
    """Parse the best epoch's validation metrics from the most recent training log CSV.

    The "best" epoch maximizes validation F-beta (the trial objective the tuner
    maximizes); recall and precision are read from that same epoch, so they describe
    the selected model. Returns ``(True, {"score", "recall", "precision"})`` on
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
            if not df.empty and "Validation Fbeta" in df.columns:
                best = df.loc[df["Validation Fbeta"].idxmax()]
                return True, {
                    "score": float(best["Validation Fbeta"]),
                    "recall": float(best.get("Validation Recall", float("nan"))),
                    "precision": float(best.get("Validation Precision", float("nan"))),
                }

    return False, {}


def sync_best_trial_artifacts(
    trials: Sequence[HPTuneTrial],
    best_trial_dir: Path,
) -> None:
    """Refresh ``best_trial/`` with the current overall best trial's ``.env`` and checkpoint.

    The ``.env`` is regenerated from the trial itself rather than copied, so the sync
    never depends on a per-trial ``.env`` file being present (it is absent for trials
    trained outside the planner, e.g. via ``run_train.sh``). A missing checkpoint no
    longer blocks the rest of the snapshot.
    """
    from model.hp_trial import TrialStatus

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
        + ["# HP-tune trial overrides (see ``HPTuneTrial.trial_env_keys``)"]
        + [f"{k}={_shell_value(v)}" for k, v in env_keys.items()]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
