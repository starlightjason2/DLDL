#!/bin/bash -l
# Submit the HPTune MPI controller job to Polaris via qsub.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_ROOT="$(readlink -f "$SCRIPT_DIR/..")"

# Load project environment
set -a
# shellcheck source=/dev/null
source "$PROJECT_ROOT/.env"
set +a

export PYTHONPATH="$PROJECT_ROOT/src"

LOG_DIR="$HPTUNE_DIR/controller_logs"
mkdir -p "$LOG_DIR"

# Pass only the vars the controller script actually needs
PASSTHROUGH_VARS=(
    "PROJECT_ROOT=$PROJECT_ROOT"
    "HPTUNE_CONTROLLER_NODES=$HPTUNE_CONTROLLER_NODES"
    "HPTUNE_TRIAL_NODES=$HPTUNE_TRIAL_NODES"
    "HPTUNE_CONTROLLER_WALLTIME=$HPTUNE_CONTROLLER_WALLTIME"
    "HPTUNE_QUEUE=$HPTUNE_QUEUE"
    "GPUS_PER_NODE=$GPUS_PER_NODE"
    "HPTUNE_MAX_TRIALS=$HPTUNE_MAX_TRIALS"
)
# Join with commas for -v
printf -v VARS '%s,' "${PASSTHROUGH_VARS[@]}"

qsub \
    -A fusiondl_aesp \
    -q "$HPTUNE_QUEUE" \
    -k doe \
    -o "$LOG_DIR/" \
    -e "$LOG_DIR/" \
    -l "select=${HPTUNE_CONTROLLER_NODES}:system=polaris:ngpus=${GPUS_PER_NODE}" \
    -l place=scatter \
    -l "walltime=$HPTUNE_CONTROLLER_WALLTIME" \
    -l filesystems=home:eagle \
    -v "${VARS%,}" \
    "$SCRIPT_DIR/controller_parallel.sh"