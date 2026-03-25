#!/bin/bash
#PBS -N dldl_train
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -q small
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -o /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/
#PBS -e /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/

set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_ROOT="$(readlink -f "$SCRIPT_DIR/..")"

set -a
# shellcheck source=/dev/null
source "$PROJECT_ROOT/.env"
set +a

cd "$PROJECT_ROOT"
# shellcheck source=/dev/null
source "$DLDL_CONDASH" && conda activate "$CONDA_ENV"

BEST_PARAMS_PATH="$PROG_DIR/${JOB_ID}_best_params.pt"
if [ -f "$BEST_PARAMS_PATH" ]; then
  echo "Existing checkpoint found at $BEST_PARAMS_PATH; skipping retraining."
else
  python src/train.py
fi

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_LOG="$PROG_DIR/train_${PBS_JOBID}.log"
echo "---- tail: $_LOG ----" && tail -n 100 "$_LOG" 2>/dev/null || true
echo "---- LIVE: tail -F $_LOG ----"
