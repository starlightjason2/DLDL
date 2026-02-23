#!/bin/bash
#PBS -N dldl_hp_ctl
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:10:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A fusiondl_aesp
# Controller for DLDL Bayesian hyperparameter optimization

set -e

# Use PROJECT_ROOT from -v (start_hptune passes it); $0 is unreliable (points to PBS spool)
PROJECT_ROOT="${PROJECT_ROOT:-$HOME}"
cd "$PROJECT_ROOT"
HPTUNE_DIR="$PROJECT_ROOT/scripts/hptune"
LOG_DIR="$HPTUNE_DIR"
CONTROLLER_SCRIPT="$HPTUNE_DIR/controller.sh"

# Bootstrap: write immediately so we know the job ran (check $HOME if project dir empty)
echo "$(date -Iseconds) JOBID=$PBS_JOBID PWD=$PWD PROJECT_ROOT=$PROJECT_ROOT" >> "${HOME}/dldl_hptune_bootstrap.log"

# Write our own log (direct redirect, no tee)
LOG_FILE="$LOG_DIR/controller_${PBS_JOBID}.log"
mkdir -p "$LOG_DIR"
exec > "$LOG_FILE" 2>&1
echo "[controller] Started at $(date), JOBID=$PBS_JOBID, LOG=$LOG_FILE"

# Chain ID propagation
if [ -n "$1" ]; then
    CHAIN_ID="$1"
    echo "[controller] Inherited CHAIN_ID: $CHAIN_ID"
else
    CHAIN_ID="chain_${PBS_JOBID}_${RANDOM}"
    echo "[controller] Generated CHAIN_ID: $CHAIN_ID"
fi

submit_worker_and_chain() {
    local trial_dir="$1"
    echo "[controller] Submitting worker for trial: $trial_dir"
    cd "$PROJECT_ROOT"
    worker_job_id=$(cd "$HPTUNE_DIR/trials/$trial_dir" && qsub -A fusiondl_aesp -q debug -l select=1:system=polaris -l place=scatter -l walltime=1:00:00 -l filesystems=home:eagle -v "PROJECT_ROOT=$PROJECT_ROOT" run.sh 2>&1 | tail -1)
    echo "[controller] Worker job: $worker_job_id"
    qsub -A fusiondl_aesp -q debug -l select=1:system=polaris -l place=scatter -l walltime=0:10:00 -l filesystems=home:eagle -v "PROJECT_ROOT=$PROJECT_ROOT" -o "$HPTUNE_DIR/controller_%j.out" -e "$HPTUNE_DIR/controller_%j.err" -W depend=afterany:$worker_job_id "$CONTROLLER_SCRIPT" "$CHAIN_ID"
}

echo "[controller] Running bayesian_hp_tuning..."
cd "$PROJECT_ROOT"
source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

# Load env so constants.py can resolve DATA_DIR, etc.
set -a
[ -f .env.polaris ] && source .env.polaris
[ -f .env ] && source .env
set +a

HPTUNE_OUTPUT=$(python -m src.bayesian_hp_tuning --chain-id "$CHAIN_ID" 2>&1)
NEXT_TRIAL=$(echo "$HPTUNE_OUTPUT" | grep "Next trial ->" | awk -F' -> ' '{print $2}')

echo "[controller] Next trial: '$NEXT_TRIAL'"

if [ -z "$NEXT_TRIAL" ]; then
    echo "[controller] ERROR: No trial from bayesian_hp_tuning. Exiting."
    echo "[controller] bayesian_hp_tuning output:"
    echo "$HPTUNE_OUTPUT"
    exit 1
fi

submit_worker_and_chain "$NEXT_TRIAL"
echo "[controller] Done."
