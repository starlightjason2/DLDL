#!/bin/bash
#PBS -N dldl_train
#PBS -l select=1:system=polaris,place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A fusiondl_aesp
#PBS -k doe

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────
set -a
# shellcheck source=/dev/null
source "$PROJECT_ROOT/.env"
# shellcheck source=/dev/null
[[ -f "${TRIAL_DIR:-}/.env" ]] && source "$TRIAL_DIR/.env"
set +a

# shellcheck source=/dev/null
source "$DLDL_CONDASH" && conda activate "$CONDA_ENV"
export PYTHONPATH="$PROJECT_ROOT/src"

# ── Distributed setup ──────────────────────────────────────────────────────
NNODES=$(wc -l < "$PBS_NODEFILE")
NRANKS_PER_NODE=${NRANKS_PER_NODE:-4}
WORLD_SIZE=$(( NNODES * NRANKS_PER_NODE ))
export WORLD_SIZE

# ── Train ──────────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

mpiexec \
    --np "$WORLD_SIZE" \
    --ppn "$NRANKS_PER_NODE" \
    --hostfile "$PBS_NODEFILE" \
    python src/train.py