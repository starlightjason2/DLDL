import types
from pathlib import Path

from helpers import load_module_from_path


def load_distributed_module():
    calls = {"init": [], "destroy": 0, "devices": []}

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        set_device=lambda device: calls["devices"].append(device)
    )

    fake_dist = types.ModuleType("torch.distributed")
    fake_dist.init_process_group = lambda **kwargs: calls["init"].append(kwargs)
    fake_dist.destroy_process_group = lambda: calls.__setitem__(
        "destroy", calls["destroy"] + 1
    )

    module = load_module_from_path(
        "test_util_distributed",
        Path(__file__).resolve().parents[1] / "src" / "util" / "distributed.py",
        injected_modules={
            "torch": fake_torch,
            "torch.distributed": fake_dist,
        },
    )
    return module, calls


def test_setup_initializes_nccl_with_env_rendezvous() -> None:
    """Initialize NCCL distributed training with the environment rendezvous method."""
    module, calls = load_distributed_module()

    module.setup(rank=2, world_size=4)

    assert calls["init"][0]["backend"] == "nccl"
    assert calls["init"][0]["init_method"] == "env://"
    assert calls["init"][0]["rank"] == 2
    assert calls["init"][0]["world_size"] == 4
    assert calls["devices"] == [0]


def test_setup_file_and_cleanup_use_file_rendezvous() -> None:
    """Initialize file-based rendezvous and tear the process group down afterward."""
    module, calls = load_distributed_module()

    module.setup_file(rank=1, world_size=3, rendezvous_file="/tmp/rdzv")
    module.cleanup()

    assert calls["init"][0]["init_method"] == "file:///tmp/rdzv"
    assert calls["devices"] == [1]
    assert calls["destroy"] == 1
