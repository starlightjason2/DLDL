#!/bin/bash
#PBS -N dldl_hptune
#PBS -l select=1:system=polaris,place=scatter,walltime=01:00:00,filesystems=home:eagle
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -j oe


set -euo pipefail

cd "$PROJECT_ROOT"

set -a
source .env
set +a

source "$DLDL_CONDASH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$PROJECT_ROOT/src"

mkdir -p "$HPTUNE_DIR/controller_logs"

exec python -m hptune_serial
