#!/bin/bash
set -e

# --- 1. Environment Setup ---
PROJECT_ROOT="$(readlink -f "$(cd "$(dirname "$0")/.." && pwd)")"
HPTUNE_SCRIPT_DIR="$PROJECT_ROOT/scripts"
HPTUNE_DATA_DIR="$PROJECT_ROOT/data/hptune"
HPTUNE_QUEUE="${HPTUNE_QUEUE:-small}"
HPTUNE_PARALLELISM="${HPTUNE_PARALLELISM:-1}"
HPTUNE_DISPATCH_SLEEP_SECONDS="${HPTUNE_DISPATCH_SLEEP_SECONDS:-60}"
HPTUNE_CONTROLLER_NODES="${HPTUNE_CONTROLLER_NODES:-}"
LOG_DIR="$HPTUNE_DATA_DIR/controller_logs"

# Serial mode keeps the original 1-node controller job.
# In prod parallel mode, the controller owns a larger allocation and launches
# one-node workers internally across that allocation.
CONTROLLER_SELECT="select=1:system=polaris,place=scatter,walltime=1:00:00,filesystems=home:eagle"
if [ "$HPTUNE_QUEUE" = "prod" ] && [ "$HPTUNE_PARALLELISM" -gt 1 ]; then
    if [ -z "$HPTUNE_CONTROLLER_NODES" ]; then
        # Default prod runs to medium unless the caller explicitly asks for a
        # larger outer allocation that routes to large.
        if [ "$HPTUNE_PARALLELISM" -ge 100 ]; then
            HPTUNE_CONTROLLER_NODES="$HPTUNE_PARALLELISM"
        else
            HPTUNE_CONTROLLER_NODES=25
        fi
    fi
    if [ "$HPTUNE_CONTROLLER_NODES" -lt "$HPTUNE_PARALLELISM" ]; then
        echo "ERROR: HPTUNE_CONTROLLER_NODES ($HPTUNE_CONTROLLER_NODES) must be >= HPTUNE_PARALLELISM ($HPTUNE_PARALLELISM)"
        exit 1
    fi
    # Match the default walltime to the routed execution queue limits.
    if [ "$HPTUNE_CONTROLLER_NODES" -ge 100 ]; then
        CONTROLLER_WALLTIME="${HPTUNE_CONTROLLER_WALLTIME:-24:00:00}"
    else
        CONTROLLER_WALLTIME="${HPTUNE_CONTROLLER_WALLTIME:-6:00:00}"
    fi
    CONTROLLER_SELECT="select=${HPTUNE_CONTROLLER_NODES}:system=polaris,place=scatter,walltime=${CONTROLLER_WALLTIME},filesystems=home:eagle"
fi

# Clean old HPTune files, but keep per-trial .env and best-checkpoint artifacts.
mkdir -p "$HPTUNE_DATA_DIR/trials" "$LOG_DIR"
find "$HPTUNE_DATA_DIR" -type f ! -name ".env" ! -name "*_best_params.pt" -delete

cd "$PROJECT_ROOT"

echo "Submitting Controller..."
FIRST_JOB_ID=$(qsub \
    -k doe \
    -o "$LOG_DIR/" \
    -e "$LOG_DIR/" \
    -q "$HPTUNE_QUEUE" \
    -l "$CONTROLLER_SELECT" \
    -v "PROJECT_ROOT=$PROJECT_ROOT,HPTUNE_QUEUE=$HPTUNE_QUEUE,HPTUNE_PARALLELISM=$HPTUNE_PARALLELISM,HPTUNE_DISPATCH_SLEEP_SECONDS=$HPTUNE_DISPATCH_SLEEP_SECONDS,HPTUNE_CONTROLLER_NODES=$HPTUNE_CONTROLLER_NODES" \
    "$HPTUNE_SCRIPT_DIR/controller.sh")

echo "Success: Controller Job ID is $FIRST_JOB_ID"
echo "Logs: $LOG_DIR/"
echo ""
echo "Watch the queue:"
echo "  watch \"qstat -u \$USER\""
echo "Watch the logs:"
echo "  watch \"ls -lt $LOG_DIR | head -8\""

echo "Use this to follow stdout once the file appears:"
echo "tail -f \"$LOG_DIR/${FIRST_JOB_ID}.OU\""