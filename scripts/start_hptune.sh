#!/bin/bash
set -e

# --- 1. Environment Setup ---
PROJECT_ROOT="$(readlink -f "$(cd "$(dirname "$0")/.." && pwd)")"
HPTUNE_SCRIPT_DIR="$PROJECT_ROOT/scripts"
HPTUNE_DATA_DIR="$PROJECT_ROOT/data/hptune"
HPTUNE_QUEUE="${HPTUNE_QUEUE:-small}"
LOG_DIR="$HPTUNE_DATA_DIR/controller_logs"

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
    -v "PROJECT_ROOT=$PROJECT_ROOT,HPTUNE_QUEUE=$HPTUNE_QUEUE" \
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