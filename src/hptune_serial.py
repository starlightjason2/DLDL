#!/usr/bin/env python3
"""Serial HP-tune: one controller step, or --trial-id to mark trials running."""

from __future__ import annotations

import argparse
import sys

from model.bayesian_hptuner import BayesianHPTuner


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--trial-id",
        nargs="+",
        metavar="TRIAL_ID",
        help="Mark one or more queued trials as running.",
    )
    args = parser.parse_args()
    tuner = BayesianHPTuner.create()

    if args.trial_id:
        tuner.mark_trials_running(args.trial_id)
        return

    tuner.run_serial()


if __name__ == "__main__":
    main()
