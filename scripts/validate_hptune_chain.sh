#!/usr/bin/env bash
# Local checks for PBS job chaining (controller.sh ↔ bayesian_hp_tuning) without submitting jobs.
# Run from repo root: bash scripts/validate_hptune_chain.sh

set -euo pipefail

_TMP_LOG="${TMPDIR:-/tmp}/validate_hptune_chain.$$.log"
exec > >(tee "$_TMP_LOG")
exec 2>&1

fail() {
  echo "FAIL: $*" >&2
  echo "---- tail: last 80 lines of this run ----" >&2
  tail -n 80 "$_TMP_LOG" >&2 || true
  rm -f "$_TMP_LOG"
  exit 1
}

echo "== 1. Parse 'Next trial -> <id>' (same grep/sed as controller.sh) =="
SAMPLE=$'2025-03-21T12:00:00 | INFO | module:fn:1 - Next trial -> trial_42\n'
NEXT_TRIAL=$(echo "$SAMPLE" | grep "Next trial ->" | tail -1 | sed 's/.*Next trial -> //' | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ "$NEXT_TRIAL" == "trial_42" ]] || fail "expected trial_42, got '$NEXT_TRIAL'"

echo "== 2. PBS job id strip for depend=afterany (numeric only) =="
WID="6972063.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"
worker_job_num="${WID%%.*}"
[[ "$worker_job_num" == "6972063" ]] || fail "expected 6972063, got '$worker_job_num'"

echo "== 3. Trial directory is a single path segment (matches qsub cwd) =="
[[ "$(basename trial_3)" == "trial_3" ]] || fail "basename"
[[ "$NEXT_TRIAL" != */* ]] || fail "trial id must not contain slashes"

echo "== 4. Python: loguru emits parseable marker =="
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=python3
command -v python3.12 >/dev/null && PY=python3.12
if ! "$PY" -c "import loguru" 2>/dev/null; then
  echo "SKIP: loguru not available for $PY (optional; Polaris conda env has it)"
else
  OUT=$("$PY" -c "
from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, format='{message}')
logger.info('Next trial -> {}', 'trial_99')
" 2>&1)
  PARSED=$(echo "$OUT" | grep "Next trial ->" | tail -1 | sed 's/.*Next trial -> //' | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  [[ "$PARSED" == "trial_99" ]] || fail "loguru parse got '$PARSED'"
fi

echo "All chain-parse checks passed."

echo ""
echo "---- tail: last 80 lines of this run ----"
tail -n 80 "$_TMP_LOG"
rm -f "$_TMP_LOG"
