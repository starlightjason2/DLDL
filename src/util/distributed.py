"""Distributed training utilities for DLDL project."""

from datetime import timedelta
import torch
import torch.distributed as dist


def setup(rank: int, world_size: int) -> None:
    """Initialize distributed training process group."""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=10),
    )
    torch.cuda.set_device(0)


def setup_file(rank: int, world_size: int, rendezvous_file: str) -> None:
    """Initialize distributed training process group using file-based rendezvous."""
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=10),
    )
    torch.cuda.set_device(rank)


def cleanup() -> None:
    """Destroy the distributed process group."""
    dist.destroy_process_group()
