#!/usr/bin/env python3
"""Serial HP-tune step: plan, train the next trial in-process, record it, then chain the next step."""

from __future__ import annotations

from model.bayesian_hp_tuner import BayesianHpTuner


def main() -> None:
    tuner = BayesianHpTuner()
    tuner.run_step()


if __name__ == "__main__":
    main()
