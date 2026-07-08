"""Model package.

Intentionally does not eagerly import submodules: importing the package used to
pull in ``bayesian_hp_tuner`` (and transitively ``util.hp_tune``), which created a
circular import. Import submodules directly, e.g. ``from model.cnn import IpCNN``.
"""
