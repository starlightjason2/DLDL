#!/bin/bash
# Start a hyperparameter-tuning run: submit the first tuning-step job.
# Each step trains one trial and then submits the next, so only one job is ever
# pending (fits Polaris `debug`: one running + one queued per user).
set -euo pipefail

# This script lives in <repo>/scripts, so the repo root is two levels up.
PROJECT_ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd "$PROJECT_ROOT"

# Load every setting from .env into the environment.
set -a
source .env
set +a

# Required settings (fail early with a clear message if missing).
: "${HPTUNE_DIR:?set it in .env}"
: "${HPTUNE_QUEUE:?set it in .env}"
: "${HPTUNE_TRAIN_WALLTIME:?set it in .env}"

# wipe the previous run
rm -rf "$HPTUNE_DIR/controller_logs/"
rm -rf "$HPTUNE_DIR/trials/"
mkdir -p "$HPTUNE_DIR/controller_logs"
mkdir -p "$HPTUNE_DIR/trials/best_trial"

echo "Log directory: $HPTUNE_DIR/controller_logs"
echo "Trial directory: $HPTUNE_DIR/trials"

STEP_JOB=$(qsub \
    -A fusiondl_aesp \
    -q "$HPTUNE_QUEUE" \
    -k doe \
    -o "$HPTUNE_DIR/controller_logs/" \
    -e "$HPTUNE_DIR/controller_logs/" \
    -l "select=1:system=polaris,place=scatter,walltime=$HPTUNE_TRAIN_WALLTIME,filesystems=home:eagle" \
    -V \
    scripts/run_hptune.sh)

echo "First step job : $STEP_JOB"
echo "Logs           : tail -f $HPTUNE_DIR/controller_logs/$STEP_JOB.OU"
