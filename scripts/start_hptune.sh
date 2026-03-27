#!/bin/bash
export PROJECT_ROOT=/lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/
cd $PROJECT_ROOT

# shellcheck source=/dev/null
source "$PROJECT_ROOT/.env"

CID=$(
  qsub \
    -k doe \
    -A fusiondl_aesp \
    -o "$HPTUNE_DIR/controller_logs/" \
    -e "$HPTUNE_DIR/controller_logs/" \
    -q "$HPTUNE_QUEUE" \
    -l "select=1:system=polaris,place=scatter,walltime=$HPTUNE_CONTROLLER_WALLTIME,filesystems=home:eagle" \
    -V  \
    "$PROJECT_ROOT/scripts/controller.sh"
) || exit 1
echo "Controller job: $CID  logs: $HPTUNE_DIR/controller_logs/"
