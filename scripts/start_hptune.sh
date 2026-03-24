#!/bin/bash
set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(readlink -f "$SCRIPT_DIR/..")}"
HPTUNE_DIR="${DLDL_HPTUNE_DIR:-$PROJECT_ROOT/data/hptune}"
HPTUNE_QUEUE="${HPTUNE_QUEUE:-small}"
HPTUNE_CONTROLLER_WALLTIME="${HPTUNE_CONTROLLER_WALLTIME:-1:00:00}"
LOG_DIR="$HPTUNE_DIR/controller_logs"
CONTROLLER_SCRIPT="$SCRIPT_DIR/controller.sh"

require_file() {
    local path="$1"
    local description="$2"
    if [ -f "$path" ]; then
        return 0
    fi
    echo "ERROR: $description not found: $path"
    exit 1
}

prepare_hptune_workspace() {
    mkdir -p "$HPTUNE_DIR/trials" "$LOG_DIR"
    find "$HPTUNE_DIR" -type f ! -name ".env" ! -name "*_best_params.pt" -delete
}

submit_controller() {
    qsub \
        -k doe \
        -o "$LOG_DIR/" \
        -e "$LOG_DIR/" \
        -q "$HPTUNE_QUEUE" \
        -l "select=1:system=polaris,place=scatter,walltime=${HPTUNE_CONTROLLER_WALLTIME},filesystems=home:eagle" \
        -v "PROJECT_ROOT=$PROJECT_ROOT,HPTUNE_QUEUE=$HPTUNE_QUEUE,DLDL_HPTUNE_DIR=$HPTUNE_DIR" \
        "$CONTROLLER_SCRIPT"
}

print_submission_summary() {
    local job_id="$1"
    local job_id_base="${job_id%%.*}"

    echo "Success: Serial controller job ID is $job_id"
    echo "Logs: $LOG_DIR/"
    echo ""
    echo "Watch the queue:"
    echo "  watch \"qstat -u \$USER\""
    echo "Watch the logs:"
    echo "  watch \"ls -lt $LOG_DIR | head -8\""
    echo ""
    echo "Follow stdout once it appears:"
    echo "  ls $LOG_DIR/${job_id_base}*.OU"
    echo "  tail -f $LOG_DIR/${job_id_base}*.OU"
}

require_file "$CONTROLLER_SCRIPT" "serial controller script"
prepare_hptune_workspace
cd "$PROJECT_ROOT"

echo "Submitting serial controller..."
FIRST_JOB_ID="$(submit_controller)"
print_submission_summary "$FIRST_JOB_ID"