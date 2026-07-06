"""Thin ``qsub`` submission helper for the serial HP-tune chain.

The chain is one self-resubmitting job (see ``BayesianHPTuner.run_step``): each job
trains one trial in-process and then submits the next step. There are never two
jobs pending at once, so it fits queues that allow only one running + one queued
job per user (e.g. Polaris ``debug``). The job id is read straight from ``qsub``
stdout — no log scraping.
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


def _select(walltime: str) -> str:
    # The serial chain runs one trial per job on a single node.
    return (
        f"select=1:system={_PBS_SYSTEM},place={_PBS_PLACE},"
        f"walltime={walltime},filesystems={_PBS_FILESYSTEMS}"
    )


def _qsub(args: list[str]) -> str:
    """Run ``qsub`` and return the job id (its sole stdout token)."""
    cmd = ["qsub", *args]
    logger.debug("qsub: {}", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"qsub failed (code {result.returncode}): {result.stderr.strip()}"
        )
    job_id = result.stdout.strip()
    if not job_id:
        raise RuntimeError(f"qsub returned no job id (stderr: {result.stderr.strip()})")
    return job_id


def _submit_step(
    *,
    log_dir: Path,
    queue: str,
    walltime: str,
    script_name: str,
) -> str:
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
            _select(walltime),
            "-V",
            f"{os.environ['PROJECT_ROOT']}/scripts/{script_name}",
        ]
    )


def submit_hptune_step(
    *,
    log_dir: Path,
    queue: str,
    walltime: str,
) -> str:
    """Submit the next HP-tune step job; returns its PBS job id."""
    return _submit_step(
        log_dir=log_dir,
        queue=queue,
        walltime=walltime,
        script_name="run_hptune.sh",
    )
