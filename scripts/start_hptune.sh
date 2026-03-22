#!/bin/bash
# Start DLDL Bayesian hyperparameter optimization chain.
# Run from project root: ./scripts/start_hptune.sh
# After qsub, always runs: tail -F on the controller combined log (Ctrl+C stops following; job keeps running).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

HPTUNE_DIR="$PROJECT_ROOT/scripts/hptune"
TRIALS_HEADER="trial_id,lr,epochs,dropout,weight_decay,batch_size,gradient_clip,lr_scheduler,lr_scheduler_factor,lr_scheduler_patience,early_stopping_patience,val_loss,status"

# Fresh chain: remove prior trial dirs and reset the CSV (same entry point as a new study).
if [[ -d "$HPTUNE_DIR/trials" ]]; then
    rm -rf "$HPTUNE_DIR/trials"
fi
mkdir -p "$HPTUNE_DIR/trials" "$HPTUNE_DIR/controller_logs"
echo "$TRIALS_HEADER" > "$HPTUNE_DIR/trials_log.csv"
echo "Reset HPTune trials: empty trials/ and fresh trials_log.csv"

# Drop stale controller logs from previous runs (combined log under controller_logs/; PBS streams next to this script).
find "$HPTUNE_DIR/controller_logs" -maxdepth 1 -name 'controller_*.txt' -type f -delete 2>/dev/null || true
find "$HPTUNE_DIR" -maxdepth 1 \( -name 'controller_*_stdout.txt' -o -name 'controller_*_stderr.txt' \) -type f -delete 2>/dev/null || true
echo "Removed prior HPTune logs: $HPTUNE_DIR/controller_logs/controller_*.txt and $HPTUNE_DIR/controller_*_{stdout,stderr}.txt"

echo "Starting HPTune chain..."
LOG_DIR="$HPTUNE_DIR"
# set -e would exit before rc=$? if qsub fails inside $(); disable -e around qsub so we always print errors.
set +e
out=$(
    qsub -A fusiondl_aesp -q debug -l select=1:system=polaris -l place=scatter -l walltime=1:00:00 -l filesystems=home:eagle -v "PROJECT_ROOT=$PROJECT_ROOT" -o "$LOG_DIR/controller_%j_stdout.txt" -e "$LOG_DIR/controller_%j_stderr.txt" "$HPTUNE_DIR/controller.sh" 2>&1
)
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
    echo "qsub failed (exit $rc):"
    printf '%s\n' "$out"
    exit 1
fi
CONTROLLER_ID=$(echo "$out" | tail -n1 | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
if [ -z "$CONTROLLER_ID" ]; then
    echo "qsub produced no job id. Full output:"
    printf '%s\n' "$out"
    exit 1
fi
CJNUM="${CONTROLLER_ID%%.*}"
echo "Controller submitted: $CONTROLLER_ID"
echo ""
echo "Monitor: qstat -u $USER"
echo "Controller job $CONTROLLER_ID log files:"
echo "  combined (stdout+stderr after exec): $LOG_DIR/controller_logs/controller_${CONTROLLER_ID}.txt"
echo "  PBS stdout (-o):                     $LOG_DIR/controller_${CJNUM}_stdout.txt"
echo "  PBS stderr (-e):                     $LOG_DIR/controller_${CJNUM}_stderr.txt"
echo "Bootstrap (if job ran): $HOME/dldl_hptune_bootstrap.log"
echo "Trials: $HPTUNE_DIR/trials/"
echo "Results: $HPTUNE_DIR/trials_log.csv"

COMBINED_LOG="$LOG_DIR/controller_logs/controller_${CONTROLLER_ID}.txt"
echo ""
echo "Live log (Ctrl+C to stop following; job keeps running): $COMBINED_LOG"
for _ in {1..120}; do
  [ -f "$COMBINED_LOG" ] && break
  sleep 1
done
tail -F "$COMBINED_LOG"
