#!/usr/bin/env python3
"""Serial HP-tune step: plan, train the next trial in-process, record it, then chain the next step."""

from __future__ import annotations

from model.bayesian_hptuner import BayesianHPTuner


def main() -> None:
    tuner = BayesianHPTuner()
    tuner.run_step()


if __name__ == "__main__":
    main()
