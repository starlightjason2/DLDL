#!/usr/bin/env python3
"""Serial HP-tune: one controller step, or ``--trial-id`` from ``run_train.sh``."""

import argparse
import sys

from model.bayesian_hptuner import BayesianHPTuner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trial-id",
        nargs="+",
        metavar="TRIAL_ID",
        help="Will start running HP tuning job with given trial_id",
    )
    args = parser.parse_args([] if argv is None else argv)
    tuner = BayesianHPTuner.create()
    if args.trial_id:
        tuner.mark_trials_running(args.trial_id)
        return
    tuner.run_serial()


if __name__ == "__main__":
    main(sys.argv[1:])
