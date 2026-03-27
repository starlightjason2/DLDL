#!/bin/bash
#PBS -N dldl_hp_ctl
#PBS -l select=1:system=polaris,place=scatter,walltime=1:00:00,filesystems=home:eagle
#PBS -A fusiondl_aesp
#PBS -k doe

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

LOG_DIR="$HPTUNE_DIR/controller_logs"
mkdir -p "$LOG_DIR"

# ── Run dispatcher ─────────────────────────────────────────────────────────
OUT=$(python -m hptune_serial 2>&1) || { echo "$OUT" >&2; exit 1; }
echo "$OUT"

# ── Parse next trial ───────────────────────────────────────────────────────
TRIAL=$(printf '%s' "$OUT" \
    | sed -n 's/^[[:space:]]*Next trial -> //p' \
    | head -1 \
    | tr -d '[:space:]')

if [[ -z "$TRIAL" ]]; then
    echo "Chain complete."
    echo "Chain $HPTUNE_CHAIN_ID finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
        >> "$LOG_DIR/chain_summary.log"
    exit 0
fi

# ── Submit trial job ───────────────────────────────────────────────────────
export TRIAL_DIR="$HPTUNE_DIR/trials/$TRIAL"
TJID=$(qsub \
    -k doe \
    -q "$HPTUNE_QUEUE" \
    -o "$LOG_DIR/" \
    -e "$LOG_DIR/" \
    -l "select=1:system=polaris,place=scatter,walltime=$HPTUNE_TRAIN_WALLTIME,filesystems=home:eagle" \
    -V \
    "$PROJECT_ROOT/scripts/run_train.sh"
)
echo "Submitted $TRIAL as $TJID"
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'),$HPTUNE_CHAIN_ID,$PBS_JOBID,$TRIAL,$TJID" \
    >> "$LOG_DIR/chain_steps.csv"

# ── Chain next controller after trial completes ────────────────────────────
NEXT=$(qsub \
    -k doe \
    -A fusiondl_aesp \
    -o "$LOG_DIR/" \
    -e "$LOG_DIR/" \
    -q "$HPTUNE_QUEUE" \
    -l "select=1:system=polaris,place=scatter,walltime=$HPTUNE_CONTROLLER_WALLTIME,filesystems=home:eagle" \
    -W "depend=afterany:$TJID" \
    -V \
    "$PROJECT_ROOT/scripts/controller.sh"
) || { echo "ERROR: failed to chain controller" >&2; exit 1; }

echo "Chained next controller as $NEXT"