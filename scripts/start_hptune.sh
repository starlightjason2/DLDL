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

mkdir -p "$HPTUNE_DIR/controller_logs"

# RESET=1 wipes the previous run, keeping only .env files and best checkpoints.
if [[ "${RESET:-0}" == "1" ]]; then
    find "$HPTUNE_DIR" -type f ! -name ".env" ! -name "*_best_params.pt" -delete
    echo "Reset: cleared $HPTUNE_DIR"
fi

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
echo "Logs           : tail $HPTUNE_DIR/controller_logs/$STEP_JOB.txt"
