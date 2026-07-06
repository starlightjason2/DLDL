#!/bin/bash
# Smoke test architecture HPTune: one short trial, no PBS chain.
#PBS -N dldl_arch_smoke
#PBS -l select=1:system=polaris,place=scatter
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -j oe

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$PROJECT_ROOT"

set -a
source .env
set +a

source "$DLDL_CONDASH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$PROJECT_ROOT/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1

export HPTUNE_MODE=architecture
export HPTUNE_DIR="${ARCH_HPTUNE_SMOKE_DIR:-$PROJECT_ROOT/data/arch_hptune_smoke}"
export HPTUNE_MAX_TRIALS=1
export NUM_EPOCHS=2
export EARLY_STOPPING_PATIENCE=0

rm -rf "$HPTUNE_DIR/trials" "$HPTUNE_DIR/controller_logs"
mkdir -p "$HPTUNE_DIR/trials/best_trial" "$HPTUNE_DIR/controller_logs"

echo "=== pytest: tests/test_arch_hptune.py ==="
python -m pytest tests/test_arch_hptune.py -q

echo "=== architecture HPTune smoke (1 trial, 2 epochs) ==="
python -m hptune_serial

python - <<'PY'
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from util.objective import PRECISION_COL, RECALL_COL, THRESHOLD_COL

root = Path(os.environ["HPTUNE_DIR"])
csv_path = root / "trials" / "trials.csv"
if not csv_path.exists():
    raise SystemExit(f"Missing trials log: {csv_path}")

df = pd.read_csv(csv_path)
if len(df) != 1:
    raise SystemExit(f"Expected 1 trial row, got {len(df)}")
if int(df.iloc[0]["status"]) != 0:
    raise SystemExit(f"Trial did not complete: status={df.iloc[0]['status']}")

for col in (
    "conv1_filters",
    "pool_size",
    "fc1_size",
    "score",
    "recall",
    "precision",
):
    if col not in df.columns or pd.isna(df.iloc[0][col]):
        raise SystemExit(f"Missing architecture trial column: {col}")

trial_id = df.iloc[0]["trial_id"]
log = root / "trials" / trial_id / f"{trial_id}_training_log.csv"
if not log.exists():
    raise SystemExit(f"Missing training log: {log}")

tlog = pd.read_csv(log)
if THRESHOLD_COL not in tlog.columns:
    raise SystemExit(f"Missing {THRESHOLD_COL} in training log")
if len(tlog) < 1:
    raise SystemExit("Training log is empty")

row = df.iloc[0]
print(
    f"Architecture smoke OK: trial={trial_id} "
    f"conv1={int(row['conv1_filters'])} pool={int(row['pool_size'])} "
    f"fc1={int(row['fc1_size'])} score={float(row['score']):.4f} "
    f"P={float(row['precision']):.4f} R={float(row['recall']):.4f}"
)
PY

echo "Architecture HPTune smoke passed."
