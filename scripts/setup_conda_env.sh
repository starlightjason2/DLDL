#!/usr/bin/env bash
# Create a clean conda env named "dldl" with Python 3.12 and NumPy 2.2.6, then pip-install this project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/soft/applications/conda/2025-09-25/mconda3}"
CONDA="${CONDA_ROOT}/bin/conda"

if [[ ! -x "$CONDA" ]]; then
  echo "Conda not found at $CONDA (set CONDA_ROOT if installed elsewhere)" >&2
  exit 1
fi

if "$CONDA" env list | grep -E '^dldl[[:space:]]' >/dev/null; then
  echo "Conda env 'dldl' already exists. Remove it first: conda env remove -n dldl" >&2
  echo "Or update: $CONDA env update -f \"$ROOT/environment.yml\" --prune" >&2
  exit 1
fi

"$CONDA" env create -f "$ROOT/environment.yml"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate dldl
export PYTHONNOUSERSITE=1
# Constrain NumPy so pip does not replace conda-forge with a mismatched wheel
pip install -e "$ROOT" -c "$ROOT/constraints.txt"

echo ""
echo "Activate with:"
echo "  source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate dldl"
echo "Verify:"
echo "  python -c \"import sys, numpy; print(sys.executable); print(numpy.__version__)\""
