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

# ── Train (single process, single GPU) ─────────────────────────────────────
cd "$PROJECT_ROOT"

python src/train.py