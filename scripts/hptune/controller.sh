#!/bin/bash
#PBS -N dldl_hp_ctl
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:10:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A fusiondl_aesp
# Controller for DLDL Bayesian hyperparameter optimization
# Note: -o/-e passed by qsub caller; without them PBS may write to $HOME

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null || cd "$(dirname "$0")")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HPTUNE_DIR="$SCRIPT_DIR"
LOG_DIR="$HPTUNE_DIR"
CONTROLLER_SCRIPT="$HPTUNE_DIR/controller.sh"

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
    worker_job_id=$(cd "$HPTUNE_DIR/trials/$trial_dir" && qsub run.sh 2>&1 | tail -1)
    echo "[controller] Worker job: $worker_job_id"
    qsub -o "$HPTUNE_DIR/controller_%j.out" -e "$HPTUNE_DIR/controller_%j.err" -W depend=afterany:$worker_job_id "$CONTROLLER_SCRIPT" "$CHAIN_ID"
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

NEXT_TRIAL=$(python -m src.bayesian_hp_tuning --chain-id "$CHAIN_ID" 2>&1 | grep "Next trial ->" | awk -F' -> ' '{print $2}')

echo "[controller] Next trial: '$NEXT_TRIAL'"

if [ -z "$NEXT_TRIAL" ]; then
    echo "[controller] ERROR: No trial from bayesian_hp_tuning. Exiting."
    exit 1
fi

submit_worker_and_chain "$NEXT_TRIAL"
echo "[controller] Done."
