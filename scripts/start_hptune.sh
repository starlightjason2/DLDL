#!/bin/bash
# Start DLDL Bayesian hyperparameter optimization chain.
# Run from project root: ./scripts/start_hptune.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$SCRIPT_DIR/hptune/trials"

echo "Starting HPTune chain..."
HPTUNE_DIR="$SCRIPT_DIR/hptune"
CONTROLLER_ID=$(qsub -o "$HPTUNE_DIR/controller_%j.out" -e "$HPTUNE_DIR/controller_%j.err" "$HPTUNE_DIR/controller.sh" 2>&1 | tail -1)
echo "Controller submitted: $CONTROLLER_ID"
echo ""
echo "Monitor: qstat -u \$USER"
echo "Logs: scripts/hptune/controller_*.out"
echo "Trials: scripts/hptune/trials/"
echo "Results: scripts/hptune/trials_log.csv"
