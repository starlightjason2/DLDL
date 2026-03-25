#!/bin/bash
#PBS -N dldl_train
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=__HPTUNE_TRAIN_WALLTIME__
#PBS -l filesystems=home:eagle
#PBS -q __HPTUNE_PBS_QUEUE__
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -o __HPTUNE_LOG_DIR__/
#PBS -e __HPTUNE_LOG_DIR__/

set -e

# Injected when materializing HP-tune trial run.sh (trial lives under TRIALS_DIR, not repo root).
PROJECT_ROOT="__DLDL_PROJECT_ROOT__"

set -a
# shellcheck source=/dev/null
source "$PROJECT_ROOT/.env"
set +a

# __HPTUNE_CD_OVERRIDE__
# shellcheck source=/dev/null
source "$DLDL_CONDASH" && conda activate "$CONDA_ENV"

# __HPTUNE_ENV_INJECT__

# __HPTUNE_POST_TRAIN_FUNCS__

BEST_PARAMS_PATH="$PROG_DIR/${JOB_ID}_best_params.pt"
if [ -f "$BEST_PARAMS_PATH" ]; then
  echo "Existing checkpoint found at $BEST_PARAMS_PATH; skipping retraining."
else
  python src/train.py
  # __HPTUNE_POST_TRAIN_RUN__
fi
