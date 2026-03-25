"""Decentralized MPI-based HPTune scheduler (no central controller)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from loguru import logger
from mpi4py import MPI

from model.bayesian_hptuner import BayesianHPTuner
from service.trial_service import TrialService
from model.hp_trial import TrialStatus

# ------------------------------------------------------------------
# MPI setup
# ------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = MPI.Get_processor_name()

hosts = comm.allgather(hostname)

GPUS_PER_NODE = int(os.environ["GPUS_PER_NODE"])
TRIAL_NODES = int(os.environ["HPTUNE_TRIAL_NODES"])

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def group_ranks() -> List[List[int]]:
    """Deterministically partition ranks into trial groups."""
    nodes = list(dict.fromkeys(hosts))  # unique hosts
    groups = []

    for i in range(0, len(nodes), TRIAL_NODES):
        chunk = nodes[i : i + TRIAL_NODES]
        if len(chunk) < TRIAL_NODES:
            continue
        group = [r for r, h in enumerate(hosts) if h in chunk]
        groups.append(group)

    return groups


GROUPS = group_ranks()


def my_group():
    for gid, g in enumerate(GROUPS):
        if rank in g:
            return gid, g
    return None, None


GROUP_ID, GROUP = my_group()


# ------------------------------------------------------------------
# Trial execution
# ------------------------------------------------------------------


def run_trial(tuner: BayesianHPTuner, trial_id: str) -> int:
    trial_dir = Path(tuner.trials_dir) / trial_id

    env = {
        **os.environ,
        "RANK": str(rank),
        "WORLD_SIZE": str(len(GROUP)),
    }

    log_path = trial_dir / f"rank_{rank}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [sys.executable, Path(tuner.project_root) / "src" / "train.py"],
        cwd=tuner.project_root,
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
        env=env,
    )

    return result.returncode


# ------------------------------------------------------------------
# Decentralized scheduler loop
# ------------------------------------------------------------------


def scheduler_loop():
    tuner = BayesianHPTuner.create()

    while True:
        # -----------------------------------
        # 1. Sync trial state
        # -----------------------------------
        trials = TrialService.get_trials()
        trials = tuner.update_trials(trials)

        # -----------------------------------
        # 2. Check termination
        # -----------------------------------
        if tuner.is_complete(trials):
            logger.info("[rank {}] all trials complete", rank)
            break

        # -----------------------------------
        # 3. Try to claim work (leader per group)
        # -----------------------------------
        if rank == GROUP[0]:
            queued = [t for t in trials if t.status == TrialStatus.QUEUED]

            if not queued:
                # propose new trial
                trials = tuner._plan_new_trials(trials)
                queued = [t for t in trials if t.status == TrialStatus.QUEUED]

            if queued:
                trial = queued[0]

                # optimistic claim
                TrialService.update_trial(
                    trial.trial_id,
                    {"status": TrialStatus.RUNNING},
                )

                trial_id = trial.trial_id
            else:
                trial_id = None
        else:
            trial_id = None

        # -----------------------------------
        # 4. Broadcast assignment within group
        # -----------------------------------
        trial_id = comm.bcast(trial_id, root=GROUP[0])

        if not trial_id:
            time.sleep(0.5)
            continue

        # -----------------------------------
        # 5. Run trial
        # -----------------------------------
        rc = run_trial(tuner, trial_id)

        # reduce result
        group_comm = comm.Create_group(comm.group.Incl(GROUP))
        group_rc = group_comm.allreduce(rc, op=MPI.MAX)

        # -----------------------------------
        # 6. Finalize trial
        # -----------------------------------
        if rank == GROUP[0]:
            if group_rc == 0:
                TrialService.update_trial(
                    trial_id,
                    {"status": TrialStatus.COMPLETED},
                )
                logger.success("trial {} complete", trial_id)
            else:
                tuner.mark_trial_failed(trial_id, return_code=int(group_rc))

        group_comm.Free()

        # small jitter to reduce contention
        time.sleep(0.1 + 0.2 * (rank % 3))


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(
        "Rank {} on {} — decentralized scheduler start",
        rank,
        hostname,
    )
    scheduler_loop()