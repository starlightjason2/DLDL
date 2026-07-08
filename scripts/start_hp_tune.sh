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
: "${HP_TUNE_DIR:?set it in .env}"
: "${HP_TUNE_QUEUE:?set it in .env}"
: "${HP_TUNE_TRAIN_WALLTIME:?set it in .env}"

if [[ "${RESET:-0}" == "1" ]]; then
  rm -rf "$HP_TUNE_DIR/controller_logs/" "$HP_TUNE_DIR/trials/"
fi

mkdir -p "$HP_TUNE_DIR/controller_logs"
mkdir -p "$HP_TUNE_DIR/trials/best_trial"

echo "Log directory: $HP_TUNE_DIR/controller_logs"
echo "Trial directory: $HP_TUNE_DIR/trials"

STEP_JOB=$(qsub \
    -A fusiondl_aesp \
    -q "$HP_TUNE_QUEUE" \
    -k doe \
    -o "$HP_TUNE_DIR/controller_logs/" \
    -e "$HP_TUNE_DIR/controller_logs/" \
    -l "select=1:system=polaris,place=scatter,walltime=$HP_TUNE_TRAIN_WALLTIME,filesystems=home:eagle" \
    -V \
    scripts/run_hp_tune.sh)

echo "First step job : $STEP_JOB"
echo "Logs           : tail -f $HP_TUNE_DIR/controller_logs/$STEP_JOB.OU"
