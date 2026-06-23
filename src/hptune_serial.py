#!/usr/bin/env python3
"""Serial HP-tune controller step.

Default: run one dispatch pass (refresh, plan, submit the next trial, chain the
next controller). ``--trial-id`` is a manual recovery hook to force trials to
``RUNNING`` without dispatching.
"""

from __future__ import annotations

import argparse

from model.bayesian_hptuner import BayesianHPTuner


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--trial-id",
        nargs="+",
        metavar="TRIAL_ID",
        help="Manually mark one or more queued trials as running (recovery only).",
    )
    args = parser.parse_args()
    tuner = BayesianHPTuner.create()

    if args.trial_id:
        tuner.mark_trials_running(args.trial_id)
        return

    tuner.dispatch_serial()


if __name__ == "__main__":
    main()
