#!/bin/bash
# Start DLDL Bayesian hyperparameter optimization chain.
# Run from project root: ./scripts/start_hptune.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$SCRIPT_DIR/hptune/trials"

echo "Starting HPTune chain..."
HPTUNE_DIR="$PROJECT_ROOT/scripts/hptune"
LOG_DIR="$HPTUNE_DIR"
CONTROLLER_ID=$(qsub -o "$LOG_DIR/controller_%j.out" -e "$LOG_DIR/controller_%j.err" "$HPTUNE_DIR/controller.sh" 2>&1 | tail -1)
echo "Controller submitted: $CONTROLLER_ID"
echo ""
echo "Monitor: qstat -u \$USER"
echo "Logs: $LOG_DIR/controller_*.out (also check \$HOME if empty)"
echo "Trials: $HPTUNE_DIR/trials/"
echo "Results: $HPTUNE_DIR/trials_log.csv"
