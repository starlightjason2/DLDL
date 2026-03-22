#!/bin/bash
#PBS -N dldl_hp_ctl
#PBS -l select=1:system=polaris,place=scatter,walltime=1:00:00,filesystems=home:eagle
#PBS -q small
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -o /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/
#PBS -e /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/
set -e

# --- 1. Setup ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME}"
HPTUNE_DIR="${DLDL_HPTUNE_DIR:-$PROJECT_ROOT/data/hptune}"
HPTUNE_QUEUE="${HPTUNE_QUEUE:-small}"
LOG_DIR="$HPTUNE_DIR/controller_logs"

cd "$PROJECT_ROOT"

source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "ERROR: required env file not found: $PROJECT_ROOT/.env"
    exit 1
fi
source "$PROJECT_ROOT/.env"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DLDL_HPTUNE_CHAIN_ID="${DLDL_HPTUNE_CHAIN_ID:-chain_${PBS_JOBID%%.*}}"

echo "======================================"
echo "Controller started"
echo "  Job ID    : $PBS_JOBID"
echo "  Chain ID  : $DLDL_HPTUNE_CHAIN_ID"
echo "  Timestamp : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "======================================"

# --- 2. Run Bayesian Optimizer ---
echo "Running Bayesian Optimization..."
OUT=$(python -m bayesian_hp_tuning 2>&1)
EXIT_CODE=$?

echo "=== Python output ==="
echo "$OUT"
echo "=== Python exit code: $EXIT_CODE ==="

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: bayesian_hp_tuning exited with code $EXIT_CODE. Aborting chain."
    exit 1
fi

# --- 3. Extract Trial ID ---
# NOTE: Python prints exactly "Next trial -> <dir_name>" to stdout.
# This grep/sed must stay in sync with _log_next_trial_marker in bayesian_hp_tuning.py.
TRIAL=$(echo "$OUT" | grep 'Next trial ->' | tail -1 | sed 's/.*Next trial -> //; s/[[:space:]]//g')

echo "=== TRIAL extracted: '$TRIAL' ==="

if [ -z "$TRIAL" ]; then
    echo "No more trials suggested by optimizer. Chain complete."
    echo "Chain $DLDL_HPTUNE_CHAIN_ID finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
        >> "$LOG_DIR/chain_summary.log"
    exit 0
fi

# --- 4. Validate Trial Directory ---
TRIAL_DIR="$HPTUNE_DIR/trials/$TRIAL"

if [ ! -d "$TRIAL_DIR" ]; then
    echo "ERROR: Trial directory does not exist: $TRIAL_DIR"
    echo "  The optimizer suggested '$TRIAL' but no matching directory was found."
    exit 1
fi

if [ ! -f "$TRIAL_DIR/run.sh" ]; then
    echo "ERROR: run.sh not found in trial directory: $TRIAL_DIR"
    exit 1
fi

# --- 5. Submit Trial Job ---
# The next controller is submitted from inside run.sh after training completes.
echo "Submitting trial job: $TRIAL"
TRIAL_JOB_ID=$(qsub \
    -A fusiondl_aesp \
    -q "$HPTUNE_QUEUE" \
    -l select=1:system=polaris,place=scatter,walltime=${HPTUNE_TRAIN_WALLTIME:-1:00:00},filesystems=home:eagle \
    -v "PROJECT_ROOT=$PROJECT_ROOT,DLDL_HPTUNE_CHAIN_ID=$DLDL_HPTUNE_CHAIN_ID,TRIAL_ID=$TRIAL,HPTUNE_QUEUE=$HPTUNE_QUEUE" \
    -k doe \
    -o "$LOG_DIR/" \
    -e "$LOG_DIR/" \
    "$TRIAL_DIR/run.sh") || {
        echo "ERROR: Trial qsub failed. Aborting chain."
        exit 1
    }

echo "======================================"
echo "Chain step complete"
echo "  Trial job : $TRIAL_JOB_ID  ($TRIAL)"
echo "  Next controller will be submitted by run.sh after training completes."
echo "======================================"

# --- 6. Log Chain Step ---
mkdir -p "$LOG_DIR"
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'),$DLDL_HPTUNE_CHAIN_ID,$PBS_JOBID,$TRIAL,$TRIAL_JOB_ID" \
    >> "$LOG_DIR/chain_steps.csv"