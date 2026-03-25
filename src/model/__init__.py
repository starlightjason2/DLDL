"""Model module: neural network models, datasets, and training functionality."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bayesian_hptuner import BayesianHPTuner
    from .cnn import IpCNN
    from .dataset import IpDataset
    from schemas.trial_schema import HPTuneTrial

__all__ = [
    "BayesianHPTuner",
    "HPTuneTrial",
    "IpDataset",
    "IpCNN",
]


def __getattr__(name: str):
    if name == "BayesianHPTuner":
        from .bayesian_hptuner import BayesianHPTuner

        return BayesianHPTuner
    if name == "HPTuneTrial":
        from schemas.trial_schema import HPTuneTrial

        return HPTuneTrial
    if name == "IpCNN":
        from .cnn import IpCNN

        return IpCNN
    if name == "IpDataset":
        from .dataset import IpDataset

        return IpDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
