#!/usr/bin/env python3
"""CLI entry for DLDL Bayesian hyperparameter tuning on Polaris."""

import argparse
import sys

from model.bayesian_hptuner import BayesianHPTuner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mark-running",
        nargs="+",
        metavar="TRIAL_ID",
        help="Mark queued trial ids as submitted/running after qsub succeeds.",
    )
    args = parser.parse_args([] if argv is None else argv)
    tuner = BayesianHPTuner()
    if args.mark_running:
        tuner.mark_trials_running(args.mark_running)
        return
    tuner.run()


if __name__ == "__main__":
    main(sys.argv[1:])
