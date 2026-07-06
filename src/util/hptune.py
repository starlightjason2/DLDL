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
import torch
from loguru import logger

from model.cnn import IpCNN
from model.trial_status import TrialStatus
from util.objective import PRECISION_COL, RECALL_COL, best_row, trial_metrics

if TYPE_CHECKING:
    from model.dataset import IpDataset
    from model.hp_trial import HPTuneTrial

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
    """Training env vars held fixed during ``HPTUNE_MODE=architecture`` runs."""
    missing = [key for key in _FIXED_TRAINING_ENV_KEYS if key not in os.environ]
    if missing:
        raise KeyError(
            "Missing training hyperparameters required for architecture HPTune: "
            + ", ".join(missing)
        )
    return {key: os.environ[key] for key in _FIXED_TRAINING_ENV_KEYS}


def fixed_training_trial_fields() -> dict[str, Any]:
    """Map fixed project training env vars to :class:`HPTuneTrial` field names."""
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
    trials: Sequence[HPTuneTrial],
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


def load_best_trial_cnn(dataset: "IpDataset") -> "IpCNN | None":
    """Build an ``IpCNN`` initialized with the global-best trial's weights.

    The network architecture is read from the environment (the same vars
    ``train.py`` uses) and the weights come from ``best_trial/*_best_params.pt``.
    Returns an eval-mode model on CPU, or ``None`` if no best checkpoint exists.
    """
    repo = Path(__file__).resolve().parents[2]
    hptune_dir = Path(os.environ["HPTUNE_DIR"])
    if not hptune_dir.is_absolute():
        hptune_dir = repo / hptune_dir

    best_dir = hptune_dir / "trials" / "best_trial"
    checkpoints = sorted(best_dir.glob("*_best_params.pt"))
    if not checkpoints:
        return None

    def _conv(prefix: str) -> tuple[int, int, int]:
        return (
            int(os.environ[f"{prefix}_FILTERS"]),
            int(os.environ[f"{prefix}_KERNEL"]),
            int(os.environ[f"{prefix}_PADDING"]),
        )

    model = IpCNN(
        dataset,
        prog_dir=str(best_dir),
        conv1=_conv("CONV1"),
        conv2=_conv("CONV2"),
        conv3=_conv("CONV3"),
        conv4=_conv("CONV4"),
        pool_size=int(os.environ["POOL_SIZE"]),
        fc1_size=int(os.environ["FC1_SIZE"]),
        fc2_size=int(os.environ["FC2_SIZE"]),
        dropout_rate=float(os.environ["DROPOUT_RATE"]),
        cls_pos_weight=float(os.environ["CLS_POS_WEIGHT"]),
        decision_threshold=float(os.environ["DECISION_THRESHOLD"]),
    )
    ckpt = torch.load(checkpoints[0], map_location="cpu")
    # Checkpoints are plain weight state_dicts; tolerate older full-state dicts too.
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


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
