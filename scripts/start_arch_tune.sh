#!/bin/bash
# Start architecture HP tune: same serial chain as training HP tune, but samples
# CNN architecture while keeping training hyperparameters fixed from .env.
set -euo pipefail

PROJECT_ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd "$PROJECT_ROOT"

set -a
source .env
set +a

: "${HP_TUNE_QUEUE:?set it in .env}"
: "${HP_TUNE_TRAIN_WALLTIME:?set it in .env}"

export HP_TUNE_MODE=architecture
export HP_TUNE_DIR="${ARCH_TUNE_DIR:-$PROJECT_ROOT/data/arch_tune}"

if [[ "${RESET:-0}" == "1" ]]; then
  rm -rf "$HP_TUNE_DIR/controller_logs/" "$HP_TUNE_DIR/trials/"
fi

mkdir -p "$HP_TUNE_DIR/controller_logs"
mkdir -p "$HP_TUNE_DIR/trials/best_trial"

echo "HP tune mode: architecture"
echo "Log directory: $HP_TUNE_DIR/controller_logs"
echo "Trial directory: $HP_TUNE_DIR/trials"
echo "Training hyperparameters are fixed from project .env"

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
