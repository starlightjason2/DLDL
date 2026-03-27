import os
import time
import json
import redis
from mpi4py import MPI

# 1. MPI & Hardware Affinity
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get local rank on node (0-3 on Polaris) to bind to specific GPU
local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
local_rank = local_comm.Get_rank()
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

# 2. Redis Connection (Point this to a node running Redis)
# Pro-tip: Launch redis-server on Rank 0 or a service node
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
r = redis.Redis(host=REDIS_HOST, port=6379, db=0)


def dummy_train_task(task_id, config):
    """Simulate a data science task (e.g., HPO trial)"""
    print(
        f"[Rank {rank}/Local {local_rank}] Starting Task {task_id} on GPU {local_rank}"
    )
    time.sleep(2)  # Simulate work
    return {"loss": 0.5, "accuracy": 0.9}


def worker_loop():
    while True:
        # 3. Atomic "Claim" - Moves task from 'todo' to a 'processing' list unique to this rank
        # This prevents task loss if the rank crashes.
        task_data = r.blmove("dldl_todo", f"processing:rank:{rank}", timeout=5)

        if task_data:
            task = json.loads(task_data)
            result = dummy_train_task(task["id"], task["config"])

            # Record result and clean up
            r.hset("dldl_results", task["id"], json.dumps(result))
            r.lrem(f"processing:rank:{rank}", 1, task_data)
            print(f"[Rank {rank}] Finished Task {task['id']}")
        else:
            # Check a global 'stop' flag or just wait
            if r.get("dldl_stop"):
                break
            time.sleep(1)


if __name__ == "__main__":
    if rank == 0:
        # Seed the queue with 20 dummy tasks
        r.delete("dldl_todo", "dldl_stop", "dldl_results")
        for i in range(20):
            r.lpush("dldl_todo", json.dumps({"id": i, "config": {"lr": 0.01}}))
        print("Queue Seeded.")

    comm.Barrier()  # Wait for Rank 0 to finish seeding
    worker_loop()
