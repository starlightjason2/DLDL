#!/bin/bash
#PBS -N dldl_hptune
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=@HPTUNE_WALLTIME@
#PBS -l filesystems=home:eagle
#PBS -q @HPTUNE_QUEUE@
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -o @HPTUNE_LOGDIR@/
#PBS -e @HPTUNE_LOGDIR@/


PROJECT_ROOT="@DLDL_ROOT@"

set -a
# shellcheck source=/dev/null
source "$PROJECT_ROOT/.env"
set +a

@DLDL_CD_AND_TRIAL_ENV@

# shellcheck source=/dev/null
source "$DLDL_CONDASH" && conda activate "$CONDA_ENV"

python -m hptune_serial --trial-id "$JOB_ID"
python src/train.py

shopt -s nullglob
rm -f "$PROG_DIR/${JOB_ID}_params_epoch"*.pt
shopt -u nullglob

mkdir -p "${HPTUNE_DIR}/controller_logs"
exec qsub -A "${HPTUNE_QSUB_ACCOUNT}" -q "${HPTUNE_QUEUE}" \
  -l "select=1:system=polaris,place=scatter,walltime=${HPTUNE_CONTROLLER_WALLTIME},filesystems=home:eagle" \
  -k doe -o "${HPTUNE_DIR}/controller_logs/" -e "${HPTUNE_DIR}/controller_logs/" \
  -v "HPTUNE_CHAIN_ID=${HPTUNE_CHAIN_ID},HPTUNE_QUEUE=${HPTUNE_QUEUE}" \
  "${PROJECT_ROOT}/scripts/controller.sh"
