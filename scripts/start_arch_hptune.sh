#!/bin/bash
# Start architecture HPTune: same serial chain as training HPTune, but samples
# CNN architecture while keeping training hyperparameters fixed from .env.
set -euo pipefail

PROJECT_ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd "$PROJECT_ROOT"

set -a
source .env
set +a

: "${HPTUNE_QUEUE:?set it in .env}"
: "${HPTUNE_TRAIN_WALLTIME:?set it in .env}"

export HPTUNE_MODE=architecture
export HPTUNE_DIR="${ARCH_HPTUNE_DIR:-$PROJECT_ROOT/data/arch_hptune}"

if [[ "${RESET:-0}" == "1" ]]; then
  rm -rf "$HPTUNE_DIR/controller_logs/" "$HPTUNE_DIR/trials/"
fi

mkdir -p "$HPTUNE_DIR/controller_logs"
mkdir -p "$HPTUNE_DIR/trials/best_trial"

echo "HPTune mode: architecture"
echo "Log directory: $HPTUNE_DIR/controller_logs"
echo "Trial directory: $HPTUNE_DIR/trials"
echo "Training hyperparameters are fixed from project .env"

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
