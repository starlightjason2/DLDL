#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$ROOT"

set -a
# shellcheck source=/dev/null
source "$ROOT/.env"
set +a

: "${HPTUNE_DIR:?not set}"
: "${HPTUNE_QUEUE:?not set}"
: "${HPTUNE_CONTROLLER_WALLTIME:?not set}"

mkdir -p "$HPTUNE_DIR/controller_logs"

if [[ "${RESET:-0}" == "1" ]]; then
    find "$HPTUNE_DIR" -type f \
        ! -name ".env" \
        ! -name "*_best_params.pt" \
        -delete
    echo "Reset: cleared $HPTUNE_DIR"
fi

CID=$(qsub \
    -k doe \
    -A fusiondl_aesp \
    -o "$HPTUNE_DIR/controller_logs/" \
    -e "$HPTUNE_DIR/controller_logs/" \
    -q "$HPTUNE_QUEUE" \
    -l "select=1:system=polaris,place=scatter,walltime=$HPTUNE_CONTROLLER_WALLTIME,filesystems=home:eagle" \
    -V \
    "$ROOT/scripts/controller.sh"
) || { echo "ERROR: qsub failed" >&2; exit 1; }

echo "Controller job : $CID"
echo "Logs           : $HPTUNE_DIR/controller_logs/"