#!/bin/bash
#PBS -N dldl_hp_ctl
#PBS -l select=1:system=polaris
#PBS -l place=scatter
# Polaris debug queue max walltime is 1h (2h is rejected: exit 188).
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A fusiondl_aesp

set -e

PROJECT_ROOT="${PROJECT_ROOT:-$HOME}"
cd "$PROJECT_ROOT"
PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)
export PROJECT_ROOT
HPTUNE="$PROJECT_ROOT/scripts/hptune"
CTRL="$HPTUNE/controller.sh"

mkdir -p "$HPTUNE/controller_logs"
# PBS -o/-e: controller_%j_{stdout,stderr}.txt (numeric %j); combined log: controller_logs/*.txt (not *.log — see below).
JNUM="${PBS_JOBID%%.*}"
# Use .txt (not .log) so repo *.log gitignore does not hide this file in the IDE.
CONTROLLER_LOG="$HPTUNE/controller_logs/controller_${PBS_JOBID}.txt"
echo "$(date -Iseconds) JOBID=$PBS_JOBID PROJECT_ROOT=$PROJECT_ROOT CONTROLLER_LOG=$CONTROLLER_LOG" >>"$HOME/dldl_hptune_bootstrap.log"
# Before exec: these lines go to PBS -o / -e for this job id; after exec, everything goes to CONTROLLER_LOG
echo "[controller] Job $PBS_JOBID log files:"
echo "[controller]   combined (stdout+stderr after exec): $CONTROLLER_LOG"
echo "[controller]   PBS stdout (-o):                     $HPTUNE/controller_${JNUM}_stdout.txt"
echo "[controller]   PBS stderr (-e):                     $HPTUNE/controller_${JNUM}_stderr.txt"
exec >"$CONTROLLER_LOG" 2>&1

cd "$PROJECT_ROOT"
source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

set -a
[ -f .env.polaris ] && . .env.polaris
[ -f .env ] && . .env
set +a

# Imports are `from model...`, `from config...` — package root is src/, not the repo root.
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

echo "[controller] bayesian_hp_tuning..."
HPTUNE_OUT=$(python -m bayesian_hp_tuning 2>&1)
echo "$HPTUNE_OUT"

NEXT_TRIAL=$(echo "$HPTUNE_OUT" | grep "Next trial ->" | tail -1 | sed 's/.*Next trial -> //' | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
echo "[controller] NEXT_TRIAL=$NEXT_TRIAL"

if [ -z "$NEXT_TRIAL" ]; then
    echo "[controller] ERROR: no Next trial line"
    exit 1
fi

echo "[controller] qsub worker ($NEXT_TRIAL)"
set +e
WOUT=$(cd "$HPTUNE/trials/$NEXT_TRIAL" && qsub -A fusiondl_aesp -q debug \
    -l select=1:system=polaris -l place=scatter -l walltime=1:00:00 -l filesystems=home:eagle \
    -v "PROJECT_ROOT=$PROJECT_ROOT" run.sh 2>&1)
WRC=$?
set -e
if [ "$WRC" -ne 0 ]; then
    echo "[controller] worker qsub failed ($WRC): $WOUT"
    exit 1
fi
WID=$(echo "$WOUT" | tail -1 | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
WN=${WID%%.*}
echo "[controller] worker job $WID"

# Polaris "generic" queue often allows only one job in Q per user. Submitting the
# worker and the dependent controller while the worker is still Q yields two in Q → qsub 38.
# Wait until the worker is no longer queued (running, held, exiting, or gone from qstat).
# Controller walltime must exceed max wait (see #PBS walltime above).
# Default wait fits in 1h controller walltime (debug queue cap) minus Python/qsub overhead.
WAIT_MAX="${HPTUNE_CHAIN_WAIT_SEC:-3000}"
WAIT_INTERVAL="${HPTUNE_CHAIN_POLL_SEC:-3}"
_elapsed=0
_st=""
echo "[controller] waiting for worker to leave Q (max ${WAIT_MAX}s, poll ${WAIT_INTERVAL}s)..."
while [ "$_elapsed" -lt "$WAIT_MAX" ]; do
  _st=$(qstat -f "$WID" 2>/dev/null | sed -n 's/^[[:space:]]*job_state = //p' | tr -d '[:space:]')
  if [ -z "$_st" ]; then
    echo "[controller] worker not in qstat (finished fast or unknown); chaining."
    break
  fi
  if [ "$_st" != "Q" ]; then
    echo "[controller] worker job_state=${_st} after ${_elapsed}s; chaining."
    break
  fi
  sleep "$WAIT_INTERVAL"
  _elapsed=$((_elapsed + WAIT_INTERVAL))
done
if [ "$_st" = "Q" ]; then
  echo "[controller] ERROR: worker still in Q after ${WAIT_MAX}s — not submitting chained controller (would exceed per-user Q limit)."
  echo "[controller] Set HPTUNE_CHAIN_WAIT_SEC (debug queue walltime is 1h max) or wait and resubmit: qsub ... -v PROJECT_ROOT=$PROJECT_ROOT ... $CTRL"
  exit 1
fi

echo "[controller] qsub next controller (depend=afterany:$WN)"
set +e
COUT=$(qsub -A fusiondl_aesp -q debug \
    -l select=1:system=polaris -l place=scatter -l walltime=1:00:00 -l filesystems=home:eagle \
    -v "PROJECT_ROOT=$PROJECT_ROOT" \
    -o "$HPTUNE/controller_%j_stdout.txt" -e "$HPTUNE/controller_%j_stderr.txt" \
    -W "depend=afterany:$WN" \
    "$CTRL" 2>&1)
CRC=$?
set -e
if [ "$CRC" -ne 0 ]; then
    echo "[controller] chain qsub failed ($CRC): $COUT"
    exit 1
fi
echo "[controller] next controller $(echo "$COUT" | tail -1 | tr -d '\r')"
echo "[controller] done"
echo "[controller] Watch worker training live: tail -F $HPTUNE/trials/$NEXT_TRIAL/train_${WID}.log"
