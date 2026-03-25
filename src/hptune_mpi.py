"""MPI-based distributed HPTune dispatcher."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from loguru import logger
from mpi4py import MPI

from model.bayesian_hptuner import BayesianHPTuner

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = MPI.Get_processor_name()
hosts_by_rank: list[str] = comm.allgather(hostname)

GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", "4"))
TRIAL_NODES = int(os.environ.get("HPTUNE_TRIAL_NODES", "1"))

_controller_host = hosts_by_rank[0]


# ------------------------------------------------------------------
# Topology helpers
# ------------------------------------------------------------------


def _unique_ordered(items: list) -> list:
    """Deduplicate while preserving insertion order."""
    seen: set = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def _worker_ranks_by_host() -> dict[str, list[int]]:
    """Map each non-controller host to its MPI ranks, validating GPU count."""
    workers: dict[str, list[int]] = {}
    for worker_rank, worker_host in enumerate(hosts_by_rank):
        if worker_host != _controller_host:
            workers.setdefault(worker_host, []).append(worker_rank)

    if not workers:
        raise RuntimeError(
            "No worker hosts available. Reserve at least one controller node "
            "plus one worker node for MPI HPTune."
        )
    for host, ranks in workers.items():
        if len(ranks) != GPUS_PER_NODE:
            raise RuntimeError(
                f"Worker host {host} has {len(ranks)} MPI ranks; "
                f"expected exactly GPUS_PER_NODE={GPUS_PER_NODE}"
            )
    return workers


def get_slots() -> list[list[int]]:
    """Group worker hosts into fixed-size trial slots of ``TRIAL_NODES`` nodes each."""
    ranks_by_host = _worker_ranks_by_host()
    hosts = list(ranks_by_host)
    slots = [
        [r for host in hosts[i : i + TRIAL_NODES] for r in ranks_by_host[host]]
        for i in range(0, len(hosts), TRIAL_NODES)
        if len(hosts[i : i + TRIAL_NODES]) == TRIAL_NODES
    ]
    if not slots:
        raise RuntimeError(
            "No full MPI trial slots available. Increase HPTUNE_CONTROLLER_NODES "
            "or decrease HPTUNE_TRIAL_NODES."
        )
    return slots


def _master_port(slot_id: int, trial_id: str) -> str:
    """Deterministic, collision-avoiding port derived from job/slot/trial identity."""
    seed = f"{os.environ.get('PBS_JOBID', 'local')}:{slot_id}:{trial_id}"
    offset = int(hashlib.sha1(seed.encode()).hexdigest()[:4], 16) % 20000
    return str(20000 + offset)


# ------------------------------------------------------------------
# Trial runner
# ------------------------------------------------------------------


def run_distributed_trial(
    tuner: BayesianHPTuner,
    *,
    trial_id: str,
    slot_id: int,
    group: list[int],
    group_comm: MPI.Comm,
) -> int:
    trial_dir = Path(tuner.trials_dir) / trial_id
    trial_env_path = trial_dir / ".env"

    if not trial_env_path.exists():
        raise FileNotFoundError(f"Missing trial env file: {trial_env_path}")

    group_hosts = [hosts_by_rank[r] for r in group]
    node_hosts = _unique_ordered(group_hosts)
    group_rank = group_comm.Get_rank()
    local_rank = sum(1 for r in group[:group_rank] if hosts_by_rank[r] == hostname)

    env = {
        **os.environ,
        **{
            k: v
            for k, v in dotenv_values(trial_env_path, encoding="utf-8").items()
            if v is not None
        },
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": f"{tuner.project_root}/src{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(
            os.pathsep
        ),
        "MASTER_ADDR": node_hosts[0],
        "MASTER_PORT": _master_port(slot_id, trial_id),
        "RANK": str(group_rank),
        "WORLD_SIZE": str(group_comm.Get_size()),
        "LOCAL_RANK": str(local_rank),
        "LOCAL_WORLD_SIZE": str(GPUS_PER_NODE),
        "NODE_RANK": str(node_hosts.index(hostname)),
    }
    for key in ("PMI_RANK", "PMI_SIZE", "PMI_LOCAL_RANK"):
        env.pop(key, None)

    log_path = trial_dir / f"dist_rank_{rank}.log"
    result = subprocess.run(
        [sys.executable, Path(tuner.project_root) / "src" / "train.py"],
        cwd=tuner.project_root,
        stdout=log_path.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    return result.returncode


# ------------------------------------------------------------------
# Controller
# ------------------------------------------------------------------


def controller() -> None:
    tuner = BayesianHPTuner.create()
    slots = get_slots()

    active_slots: dict[int, str] = {}  # slot_id -> trial_id
    logger.debug(
        "[controller] host={} slots={} trial_nodes={} gpus_per_node={}",
        _controller_host,
        len(slots),
        TRIAL_NODES,
        GPUS_PER_NODE,
    )

    while True:
        while comm.Iprobe(source=MPI.ANY_SOURCE):
            msg = comm.recv(source=MPI.ANY_SOURCE)
            slot_id, trial_id = msg["slot"], msg["trial"]

            if msg["type"] == "DONE":
                logger.success("[controller] DONE  trial={} slot={}", trial_id, slot_id)
                active_slots.pop(slot_id, None)
            elif msg["type"] == "FAILED":
                logger.error(
                    "[controller] FAILED trial={} slot={} rc={}",
                    trial_id,
                    slot_id,
                    msg["return_code"],
                )
                tuner.mark_trial_failed(trial_id, return_code=int(msg["return_code"]))
                active_slots.pop(slot_id, None)

        trials = tuner.plan_and_enqueue()
        queued = tuner.get_dispatchable_trials(trials)
        free_slots = [sid for sid in range(len(slots)) if sid not in active_slots]
        assignments = list(zip(free_slots, queued[: len(free_slots)]))

        if assignments:
            tuner.mark_trials_running([tid for _, tid in assignments])
            for slot_id, trial_id in assignments:
                slot_group = slots[slot_id]
                for worker_rank in slot_group:
                    comm.send(
                        {
                            "type": "TASK",
                            "trial": trial_id,
                            "slot": slot_id,
                            "group": slot_group,
                        },
                        dest=worker_rank,
                    )
                active_slots[slot_id] = trial_id
                logger.info(
                    "[controller] DISPATCH trial={} slot={} ranks={}",
                    trial_id,
                    slot_id,
                    slot_group,
                )

        if tuner.is_complete(trials) and not active_slots:
            for worker_rank in range(1, size):
                comm.send({"type": "STOP"}, dest=worker_rank)
            break

        time.sleep(1)


# ------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------


def worker() -> None:
    tuner = BayesianHPTuner.create()

    while True:
        msg = comm.recv(source=0)

        if msg["type"] == "STOP":
            break
        if msg["type"] != "TASK":
            raise RuntimeError(f"Unknown MPI task message: {msg}")

        trial_id, slot_id, group = msg["trial"], msg["slot"], msg["group"]
        group_comm = comm.Create_group(comm.group.Incl(group))

        try:
            if group_comm == MPI.COMM_NULL:
                continue

            rc = run_distributed_trial(
                tuner,
                trial_id=trial_id,
                slot_id=slot_id,
                group=group,
                group_comm=group_comm,
            )
            group_rc = group_comm.allreduce(rc, op=MPI.MAX)

            if group_comm.rank == 0:
                comm.send(
                    {
                        "type": "DONE" if group_rc == 0 else "FAILED",
                        "trial": trial_id,
                        "slot": slot_id,
                        "return_code": int(group_rc),
                    },
                    dest=0,
                )
        finally:
            if group_comm != MPI.COMM_NULL:
                group_comm.Free()


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(
        "Rank {} on {} (size={}) — starting as {}",
        rank,
        hostname,
        size,
        "controller" if rank == 0 else "worker",
    )
    controller() if rank == 0 else worker()
