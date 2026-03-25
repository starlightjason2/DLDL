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

# __HPTUNE_CD_OVERRIDE__
# shellcheck source=/dev/null
source "$DLDL_CONDASH" && conda activate "$CONDA_ENV"

# __HPTUNE_ENV_INJECT__

BEST_PARAMS_PATH="$PROG_DIR/${JOB_ID}_best_params.pt"
if [ -f "$BEST_PARAMS_PATH" ]; then
  echo "Existing checkpoint found at $BEST_PARAMS_PATH; skipping retraining."
else
  python src/train.py
fi
