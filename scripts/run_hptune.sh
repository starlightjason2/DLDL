#!/bin/bash
#PBS -N dldl_hptune
#PBS -l select=1:system=polaris,place=scatter,walltime=01:00:00,filesystems=home:eagle
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -j oe


set -euo pipefail

cd "$PROJECT_ROOT"

# Keep qsub -V overrides (e.g. architecture mode and ARCH_HPTUNE_DIR).
_saved_hptune_dir="${HPTUNE_DIR-}"
_saved_hptune_mode="${HPTUNE_MODE-}"

set -a
source .env
set +a

if [[ -n "$_saved_hptune_dir" ]]; then
  export HPTUNE_DIR="$_saved_hptune_dir"
fi
if [[ -n "$_saved_hptune_mode" ]]; then
  export HPTUNE_MODE="$_saved_hptune_mode"
fi

source "$DLDL_CONDASH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$PROJECT_ROOT/src"

mkdir -p "$HPTUNE_DIR/controller_logs"

exec python -m hptune_serial
