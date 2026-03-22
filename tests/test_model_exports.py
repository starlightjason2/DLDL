from model import HPTuneTrial


def test_model_exports_hptune_trial() -> None:
    """Verify the lazy model package export exposes ``HPTuneTrial``."""
    assert HPTuneTrial.__name__ == "HPTuneTrial"
