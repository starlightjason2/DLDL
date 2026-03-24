#!/bin/bash
set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(readlink -f "$SCRIPT_DIR/..")"

HPTUNE_DIR="${DLDL_HPTUNE_DIR:-$PROJECT_ROOT/data/hptune}"
LOG_DIR="$HPTUNE_DIR/controller_logs"
CONTROLLER_SCRIPT="$SCRIPT_DIR/controller_parallel.sh"
HPTUNE_CONTROLLER_NODES="${HPTUNE_CONTROLLER_NODES:-4}"
HPTUNE_TRIAL_NODES="${HPTUNE_TRIAL_NODES:-1}"
HPTUNE_CONTROLLER_WALLTIME="${HPTUNE_CONTROLLER_WALLTIME:-3:00:00}"
HPTUNE_QUEUE="${HPTUNE_QUEUE:-prod}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
HPTUNE_MAX_TRIALS="${HPTUNE_MAX_TRIALS:-10}"

mkdir -p "$LOG_DIR"

# PBS Pro: place / walltime / filesystems are separate from the select chunk (see scripts/run_train.sh).
# Jamming them into one -l string can yield wrong or zero GPU assignment.
qsub \
  -A fusiondl_aesp \
  -q "$HPTUNE_QUEUE" \
  -k doe \
  -o "$LOG_DIR/" \
  -e "$LOG_DIR/" \
  -l "select=${HPTUNE_CONTROLLER_NODES}:system=polaris:ngpus=${GPUS_PER_NODE}" \
  -l place=scatter \
  -l "walltime=${HPTUNE_CONTROLLER_WALLTIME}" \
  -l filesystems=home:eagle \
  -v "PROJECT_ROOT=$PROJECT_ROOT,DLDL_HPTUNE_DIR=$HPTUNE_DIR,HPTUNE_CONTROLLER_NODES=$HPTUNE_CONTROLLER_NODES,HPTUNE_TRIAL_NODES=$HPTUNE_TRIAL_NODES,HPTUNE_CONTROLLER_WALLTIME=$HPTUNE_CONTROLLER_WALLTIME,HPTUNE_QUEUE=$HPTUNE_QUEUE,GPUS_PER_NODE=$GPUS_PER_NODE,HPTUNE_MAX_TRIALS=$HPTUNE_MAX_TRIALS" \
  "$CONTROLLER_SCRIPT"