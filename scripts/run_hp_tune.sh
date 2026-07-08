#!/bin/bash
#PBS -N dldl_hp_tune
#PBS -l select=1:system=polaris,place=scatter,walltime=01:00:00,filesystems=home:eagle
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -j oe


set -euo pipefail

cd "$PROJECT_ROOT"

# Keep qsub -V overrides (e.g. architecture mode and ARCH_TUNE_DIR).
_saved_hp_tune_dir="${HP_TUNE_DIR-}"
_saved_hp_tune_mode="${HP_TUNE_MODE-}"

set -a
source .env
set +a

if [[ -n "$_saved_hp_tune_dir" ]]; then
  export HP_TUNE_DIR="$_saved_hp_tune_dir"
fi
if [[ -n "$_saved_hp_tune_mode" ]]; then
  export HP_TUNE_MODE="$_saved_hp_tune_mode"
fi

source "$DLDL_CONDASH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$PROJECT_ROOT/src"

mkdir -p "$HP_TUNE_DIR/controller_logs"

exec python -m hp_tune_serial
