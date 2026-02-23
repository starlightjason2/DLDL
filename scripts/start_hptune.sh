#!/bin/bash
# Start DLDL Bayesian hyperparameter optimization chain.
# Run from project root: ./scripts/start_hptune.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$SCRIPT_DIR/hptune/trials"
HPTUNE_DIR="$PROJECT_ROOT/scripts/hptune"
[ -f "$HPTUNE_DIR/trials_log.csv" ] || echo "lr,epochs,dropout,val_loss,status" > "$HPTUNE_DIR/trials_log.csv"

echo "Starting HPTune chain..."
LOG_DIR="$HPTUNE_DIR"
CONTROLLER_ID=$(qsub -A fusiondl_aesp -q debug -l select=1:system=polaris -l place=scatter -l walltime=0:10:00 -l filesystems=home:eagle -v "PROJECT_ROOT=$PROJECT_ROOT" -o "$LOG_DIR/controller_%j.out" -e "$LOG_DIR/controller_%j.err" "$HPTUNE_DIR/controller.sh" 2>&1 | tail -1)
echo "Controller submitted: $CONTROLLER_ID"
echo ""
echo "Monitor: qstat -u \$USER"
echo "Logs: $LOG_DIR/controller_<jobid>.log"
echo "Bootstrap (if job ran): ~/dldl_hptune_bootstrap.log"
echo "Trials: $HPTUNE_DIR/trials/"
echo "Results: $HPTUNE_DIR/trials_log.csv"
