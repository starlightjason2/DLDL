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
CONTROLLER_SCRIPT="$SCRIPT_DIR/controller.sh"

prepare_hptune_workspace() {
  mkdir -p "$HPTUNE_DIR/trials" "$LOG_DIR"
  find "$HPTUNE_DIR" -type f ! -name ".env" ! -name "*_best_params.pt" -delete
}

submit_controller() {
  qsub -k doe -o "$LOG_DIR/" -e "$LOG_DIR/" -q "$HPTUNE_QUEUE" \
    -l "select=1:system=polaris,place=scatter,walltime=$HPTUNE_CONTROLLER_WALLTIME,filesystems=home:eagle" \
    -v "PROJECT_ROOT=$PROJECT_ROOT,HPTUNE_QUEUE=$HPTUNE_QUEUE" \
    "$CONTROLLER_SCRIPT"
}

prepare_hptune_workspace
cd "$PROJECT_ROOT"

echo "Submitting serial controller..."
FIRST_JOB_ID="$(submit_controller)"
echo "Success: job $FIRST_JOB_ID  Logs: $LOG_DIR/  watch: qstat -u \$USER  tail -f $LOG_DIR/${FIRST_JOB_ID%%.*}*.OU"
