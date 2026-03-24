"""PyTorch distributed helpers (optional; training may call ``dist`` APIs directly)."""


def setup(*_args, **_kwargs) -> None:
    """Reserved hook for process-group setup; CNN may use ``dist.init_process_group`` instead."""
    return None


def cleanup() -> None:
    """Reserved hook for teardown; CNN may use ``dist.destroy_process_group`` instead."""
    return None
