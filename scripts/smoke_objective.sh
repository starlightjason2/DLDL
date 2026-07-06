#!/bin/bash
# Smoke test objective/threshold tuning before a full training run.
#PBS -N dldl_smoke
#PBS -l select=1:system=polaris,place=scatter
#PBS -l walltime=00:10:00
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

echo "=== pytest: tests/test_objective.py ==="
python -m pytest tests/test_objective.py -q

echo "=== dev-set checkpoint smoke ==="
python scripts/smoke_objective.py

echo "=== 2-epoch training smoke (JOB_ID=smoke_prec90) ==="
export JOB_ID=smoke_prec90
export NUM_EPOCHS=2
export EARLY_STOPPING_PATIENCE=0
python src/train.py

python - <<'PY'
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from util.objective import PRECISION_COL, RECALL_COL, THRESHOLD_COL, min_precision

prog = Path(os.environ["PROG_DIR"])
log = prog / f"{os.environ['JOB_ID']}_training_log.csv"
if not log.exists():
    raise SystemExit(f"Missing training log: {log}")

df = pd.read_csv(log)
if THRESHOLD_COL not in df.columns:
    raise SystemExit(f"Missing {THRESHOLD_COL} column")

for _, row in df.iterrows():
    threshold = float(row[THRESHOLD_COL])
    precision = float(row[PRECISION_COL])
    recall = float(row[RECALL_COL])
    if not (0.01 <= threshold <= 0.99):
        raise SystemExit(f"Bad threshold {threshold} at epoch {row['epoch']}")
    if recall == 1.0 and precision < 0.5:
        raise SystemExit(
            f"Degenerate metrics at epoch {row['epoch']}: P={precision} R={recall}"
        )

best = df.loc[df[PRECISION_COL].idxmax()]
print(
    f"Training smoke OK: best logged precision={best[PRECISION_COL]:.4f}, "
    f"recall={best[RECALL_COL]:.4f}, threshold={best[THRESHOLD_COL]:.4f}, "
    f"floor={min_precision():.2f}"
)
PY

echo "All smoke tests passed."
