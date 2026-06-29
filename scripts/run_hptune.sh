#!/bin/bash
#PBS -N dldl_hptune
#PBS -l select=1:system=polaris,place=scatter,walltime=01:00:00,filesystems=home:eagle
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -j oe


set -euo pipefail

cd "$PROJECT_ROOT"

# Preserve a chain id provided by the launcher / previous step across the .env
# reload (.env carries only a placeholder; the real id is generated once in
# start_hptune.sh and propagated to each step via `qsub -V`).
_CHAIN_ID="${HPTUNE_CHAIN_ID:-}"

set -a
source .env
set +a

[[ -n "$_CHAIN_ID" ]] && export HPTUNE_CHAIN_ID="$_CHAIN_ID"

source "$DLDL_CONDASH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$PROJECT_ROOT/src"

mkdir -p "$HPTUNE_DIR/controller_logs"

exec python -m hptune_serial
