import hashlib
import os
import subprocess
import sys
import time

from dotenv import dotenv_values
from mpi4py import MPI

from model.bayesian_hptuner import BayesianHPTuner

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = MPI.Get_processor_name()
hosts_by_rank = comm.allgather(hostname)

GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", "4"))
TRIAL_NODES = int(os.environ.get("HPTUNE_TRIAL_NODES", "1"))


def _unique_preserving_order(items):
    unique_items = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def _controller_host():
    return hosts_by_rank[0]


def _worker_ranks_by_host():
    worker_hosts = {}
    for worker_rank, worker_host in enumerate(hosts_by_rank):
        if worker_host == _controller_host():
            continue
        worker_hosts.setdefault(worker_host, []).append(worker_rank)

    if not worker_hosts:
        raise RuntimeError(
            "No worker hosts are available. Reserve at least one controller node "
            "plus one worker node for MPI HPTune."
        )

    for worker_host, worker_ranks in worker_hosts.items():
        if len(worker_ranks) != GPUS_PER_NODE:
            raise RuntimeError(
                f"Worker host {worker_host} has {len(worker_ranks)} MPI ranks; "
                f"expected exactly GPUS_PER_NODE={GPUS_PER_NODE}"
            )

    return worker_hosts


def get_slots():
    worker_ranks_by_host = _worker_ranks_by_host()
    worker_hosts = list(worker_ranks_by_host.keys())
    slots = []

    for i in range(0, len(worker_hosts), TRIAL_NODES):
        slot_hosts = worker_hosts[i : i + TRIAL_NODES]
        if len(slot_hosts) != TRIAL_NODES:
            continue
        slot_ranks = []
        for slot_host in slot_hosts:
            slot_ranks.extend(worker_ranks_by_host[slot_host])
        slots.append(slot_ranks)

    return slots


def _master_port(slot_id, trial_id):
    pbs_job_id = os.environ.get("PBS_JOBID", "local")
    seed = f"{pbs_job_id}:{slot_id}:{trial_id}"
    offset = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:4], 16) % 20000
    return str(20000 + offset)


def controller():
    tuner = BayesianHPTuner()
    slots = get_slots()
    active = {}  # slot_id -> trial_id

    if not slots:
        raise RuntimeError(
            "No full MPI trial slots are available. Increase HPTUNE_CONTROLLER_NODES "
            "or decrease HPTUNE_TRIAL_NODES."
        )

    print(
        f"[controller] controller_host={_controller_host()} slots={len(slots)} "
        f"trial_nodes={TRIAL_NODES} gpus_per_node={GPUS_PER_NODE}",
        flush=True,
    )

    while True:
        while comm.Iprobe(source=MPI.ANY_SOURCE):
            msg = comm.recv(source=MPI.ANY_SOURCE)
            slot_id = msg["slot"]
            trial_id = msg["trial"]

            if msg["type"] == "DONE":
                print(f"[controller] DONE trial={trial_id} slot={slot_id}", flush=True)
                active.pop(slot_id, None)
            elif msg["type"] == "FAILED":
                print(
                    f"[controller] FAILED trial={trial_id} slot={slot_id} rc={msg['return_code']}",
                    flush=True,
                )
                tuner.mark_trial_failed(trial_id, return_code=int(msg["return_code"]))
                active.pop(slot_id, None)

        trials = tuner.sync_and_load()
        trials, _ = tuner.plan_and_enqueue()
        queued = tuner.get_dispatchable_trials(trials)

        free_slots = [slot_id for slot_id in range(len(slots)) if slot_id not in active]
        assignments = list(zip(free_slots, queued[: len(free_slots)]))

        if assignments:
            trial_ids = [trial_id for _, trial_id in assignments]
            tuner.mark_trials_running(trial_ids)

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
                active[slot_id] = trial_id
                print(
                    f"[controller] DISPATCH trial={trial_id} slot={slot_id} ranks={slot_group}",
                    flush=True,
                )

        if tuner.is_complete(trials) and not active:
            for worker_rank in range(1, size):
                comm.send({"type": "STOP"}, dest=worker_rank)
            break

        time.sleep(1)


def worker():
    tuner = BayesianHPTuner()

    while True:
        msg = comm.recv(source=0)

        if msg["type"] == "STOP":
            break

        if msg["type"] != "TASK":
            raise RuntimeError(f"Unknown MPI task message: {msg}")

        trial_id = msg["trial"]
        slot_id = msg["slot"]
        group = msg["group"]
        group_comm = comm.Create_group(comm.group.Incl(group))

        try:
            if group_comm == MPI.COMM_NULL:
                continue

            return_code = run_distributed_trial(
                tuner=tuner,
                trial_id=trial_id,
                slot_id=slot_id,
                group=group,
                group_comm=group_comm,
            )
            group_return_code = group_comm.allreduce(return_code, op=MPI.MAX)

            if group_comm.rank == 0:
                msg_type = "DONE" if group_return_code == 0 else "FAILED"
                comm.send(
                    {
                        "type": msg_type,
                        "trial": trial_id,
                        "slot": slot_id,
                        "return_code": int(group_return_code),
                    },
                    dest=0,
                )
        finally:
            if group_comm != MPI.COMM_NULL:
                group_comm.Free()


def run_distributed_trial(tuner, trial_id, slot_id, group, group_comm):
    trial_dir = os.path.join(tuner.trials_dir, trial_id)
    trial_env_path = os.path.join(trial_dir, ".env")

    if not os.path.exists(trial_env_path):
        raise FileNotFoundError(f"Missing trial env file: {trial_env_path}")

    group_hosts = [hosts_by_rank[group_rank] for group_rank in group]
    node_hosts = _unique_preserving_order(group_hosts)
    group_rank = group_comm.Get_rank()
    world_size = group_comm.Get_size()
    local_rank = sum(
        1
        for group_rank_id in group[:group_rank]
        if hosts_by_rank[group_rank_id] == hostname
    )
    node_rank = node_hosts.index(hostname)

    env = os.environ.copy()
    env.update(
        {
            key: value
            for key, value in dotenv_values(trial_env_path).items()
            if value is not None
        }
    )
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = (
        f"{tuner.project_root}/src{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(
            os.pathsep
        )
    )
    env["MASTER_ADDR"] = node_hosts[0]
    env["MASTER_PORT"] = _master_port(slot_id, trial_id)
    env["RANK"] = str(group_rank)
    env["WORLD_SIZE"] = str(world_size)
    env["LOCAL_RANK"] = str(local_rank)
    env["LOCAL_WORLD_SIZE"] = str(GPUS_PER_NODE)
    env["NODE_RANK"] = str(node_rank)
    env.pop("PMI_RANK", None)
    env.pop("PMI_SIZE", None)
    env.pop("PMI_LOCAL_RANK", None)

    command = [sys.executable, os.path.join(tuner.project_root, "src", "train.py")]
    log_path = os.path.join(trial_dir, f"dist_rank_{rank}.log")

    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=tuner.project_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    return int(result.returncode)


if __name__ == "__main__":
    if rank == 0:
        controller()
    else:
        worker()
