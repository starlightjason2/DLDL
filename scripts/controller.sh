#!/bin/bash
#PBS -N dldl_hp_ctl
#PBS -l select=1:system=polaris,place=scatter,walltime=1:00:00,filesystems=home:eagle
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -o /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/
#PBS -e /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/

# ── Environment ───────────────────────────────────────────────────────────
# shellcheck source=/dev/null
cd "$PROJECT_ROOT"
source "$DLDL_CONDASH" && conda activate "$CONDA_ENV"
source "$PROJECT_ROOT/.env"


# ── Run dispatcher ────────────────────────────────────────────────────────
OUT=$(python -m hptune_serial 2>&1) || { echo "$OUT" >&2; exit 1; }
echo "$OUT"

# ── Parse next trial ──────────────────────────────────────────────────────
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

# ── Submit trial job ──────────────────────────────────────────────────────
TJID=$(qsub \
    -k doe \
    -q "$HPTUNE_QUEUE" \
    -o "$LOG_DIR/" \
    -e "$LOG_DIR/" \
    -l "select=1:system=polaris,place=scatter,walltime=$HPTUNE_TRAIN_WALLTIME,filesystems=home:eagle" \
    -V \
    "$HPTUNE_DIR/trials/$TRIAL/run.sh"
) || { echo "ERROR: qsub failed for $TRIAL" >&2; exit 1; }

echo "Submitted $TRIAL as $TJID"
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'),$HPTUNE_CHAIN_ID,$PBS_JOBID,$TRIAL,$TJID" \
    >> "$LOG_DIR/chain_steps.csv"