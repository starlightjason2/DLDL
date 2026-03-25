#!/bin/bash
#PBS -N dldl_preprocess
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -q small
#PBS -A fusiondl_aesp
#PBS -o /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/processed_data
#PBS -e /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/processed_data

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

python src/preprocess_data.py

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "---- tail PBS stdout (last 100 lines): ${_REPO}/preprocess_${PBS_JOBID}.out ----"
tail -n 100 "${_REPO}/preprocess_${PBS_JOBID}.out" 2>/dev/null || true
echo "---- LIVE follow: tail -F ${_REPO}/preprocess_${PBS_JOBID}.out ----"
