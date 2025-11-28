#!/usr/bin/env bash
set -euo pipefail

LOGDIR=${1:-runs}
PORT=${PORT:-6006}

echo "Starting TensorBoard on port ${PORT} for logdir ${LOGDIR}" 
exec tensorboard --logdir "${LOGDIR}" --port "${PORT}" --reload_interval 5
