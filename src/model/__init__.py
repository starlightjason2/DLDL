"""Model package.

Intentionally does not eagerly import submodules: importing the package used to
pull in ``bayesian_hptuner`` (and transitively ``util.hptune``), which created a
circular import. Import submodules directly, e.g. ``from model.cnn import IpCNN``.
"""
