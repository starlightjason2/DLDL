#!/usr/bin/env python3
"""
CLI entry for DLDL Bayesian hyperparameter tuning on Polaris.

Workflow:
    1) Load or initialize trials_log.csv
    2) Update in-progress trials by parsing training log CSVs
    3) If completed trials < NUM_INITIAL_TRIALS, sample randomly; else use BO
    4) Create trial dir with run.sh and .env, print trial name for controller
"""

from model.bayesian_hptuner import BayesianHPTuner


def main() -> None:
    BayesianHPTuner().run()


if __name__ == "__main__":
    main()
