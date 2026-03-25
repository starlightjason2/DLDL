#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_ROOT="$(readlink -f "$SCRIPT_DIR/..")"

set -a
# shellcheck source=/dev/null
source "$PROJECT_ROOT/.env"
set +a

export PYTHONPATH="$PROJECT_ROOT/src"
LOG_DIR="$HPTUNE_DIR/controller_logs"
CONTROLLER_SCRIPT="$SCRIPT_DIR/controller_parallel.sh"

mkdir -p "$LOG_DIR"

qsub -A fusiondl_aesp -q "$HPTUNE_QUEUE" -k doe -o "$LOG_DIR/" -e "$LOG_DIR/" \
  -l "select=${HPTUNE_CONTROLLER_NODES}:system=polaris:ngpus=${GPUS_PER_NODE}" -l place=scatter \
  -l "walltime=$HPTUNE_CONTROLLER_WALLTIME" -l filesystems=home:eagle \
  -v "PROJECT_ROOT=$PROJECT_ROOT,HPTUNE_CONTROLLER_NODES=$HPTUNE_CONTROLLER_NODES,HPTUNE_TRIAL_NODES=$HPTUNE_TRIAL_NODES,HPTUNE_CONTROLLER_WALLTIME=$HPTUNE_CONTROLLER_WALLTIME,HPTUNE_QUEUE=$HPTUNE_QUEUE,GPUS_PER_NODE=$GPUS_PER_NODE,HPTUNE_MAX_TRIALS=$HPTUNE_MAX_TRIALS" \
  "$CONTROLLER_SCRIPT"
