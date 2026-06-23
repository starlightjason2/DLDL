#!/bin/bash
#PBS -N dldl_hp_ctl
#PBS -l select=1:system=polaris,place=scatter,walltime=1:00:00,filesystems=home:eagle
#PBS -A fusiondl_aesp
#PBS -k doe

# Thin launcher: all dispatch + chaining logic lives in `hptune_serial`
# (BayesianHPTuner.dispatch_serial). This script only sets up the environment.
set -euo pipefail

cd "$PROJECT_ROOT"

# ── Environment ────────────────────────────────────────────────────────────
set -a
# shellcheck source=/dev/null
source "$PROJECT_ROOT/.env"
set +a

# shellcheck source=/dev/null
source "$DLDL_CONDASH" && conda activate "$CONDA_ENV"
export PYTHONPATH="$PROJECT_ROOT/src"

mkdir -p "$HPTUNE_DIR/controller_logs"

# Plans the next trial, submits its training job, and chains the next
# controller (depend=afterany) — or exits cleanly when the chain is complete.
exec python -m hptune_serial
