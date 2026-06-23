"""Thin ``qsub`` submission helpers for the serial HP-tune controller chain.

All PBS orchestration lives here (and in ``BayesianHPTuner.dispatch_serial``) so the
shell scripts stay dumb launchers and the chaining logic stays unit-testable: the
trial id is passed in directly and the job id is read straight from ``qsub`` stdout —
no log-line scraping.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from loguru import logger

# Polaris resource template. Edit here if the system / filesystems change.
_PBS_SYSTEM = "polaris"
_PBS_PLACE = "scatter"
_PBS_FILESYSTEMS = "home:eagle"


def _select(nodes: int, walltime: str) -> str:
    return (
        f"select={nodes}:system={_PBS_SYSTEM},place={_PBS_PLACE},"
        f"walltime={walltime},filesystems={_PBS_FILESYSTEMS}"
    )


def _qsub(args: list[str], *, env: dict[str, str] | None = None) -> str:
    """Run ``qsub`` and return the job id (its sole stdout token)."""
    cmd = ["qsub", *args]
    logger.debug("qsub: {}", " ".join(cmd))
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"qsub failed (code {result.returncode}): {result.stderr.strip()}"
        )
    job_id = result.stdout.strip()
    if not job_id:
        raise RuntimeError(f"qsub returned no job id (stderr: {result.stderr.strip()})")
    return job_id


def submit_trial(
    *,
    trial_dir: Path,
    log_dir: Path,
    nodes: int,
    queue: str,
    walltime: str,
) -> str:
    """Submit the training job for one trial; returns its PBS job id.

    ``TRIAL_DIR`` is exported into the job's environment (via ``-V``) so
    ``run_train.sh`` can source the trial's per-trial ``.env`` overrides.
    """
    project_root = os.environ["PROJECT_ROOT"]
    env = {**os.environ, "TRIAL_DIR": str(trial_dir)}
    return _qsub(
        [
            "-k",
            "doe",
            "-q",
            queue,
            "-o",
            f"{log_dir}/",
            "-e",
            f"{log_dir}/",
            "-l",
            _select(nodes, walltime),
            "-V",
            f"{project_root}/scripts/run_train.sh",
        ],
        env=env,
    )


def submit_controller(
    *,
    depend_after: str,
    log_dir: Path,
    nodes: int,
    queue: str,
    walltime: str,
) -> str:
    """Chain the next controller, gated on ``depend_after`` finishing (any exit)."""
    project_root = os.environ["PROJECT_ROOT"]
    return _qsub(
        [
            "-k",
            "doe",
            "-q",
            queue,
            "-o",
            f"{log_dir}/",
            "-e",
            f"{log_dir}/",
            "-l",
            _select(nodes, walltime),
            "-W",
            f"depend=afterany:{depend_after}",
            "-V",
            f"{project_root}/scripts/controller.sh",
        ]
    )
