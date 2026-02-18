#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default stack configuration (no command-line arguments required).
QCAR_ID=0
VIRTUAL_PORT_STRIDE=10
NODES="10,2,4,14,20,22,9,7,0"
CRUISE_SPEED=0.65
SAMPLE_RATE=100
DETECTION_PORT=18666
DEPTH_PORT=18777
INFERENCE_MODE="custom"

if (($# > 0)); then
  echo "run_competition_stack.sh takes no arguments. Edit defaults inside the script if needed."
  exit 1
fi

VIDEO3D_PORT=$((18965 + QCAR_ID * VIRTUAL_PORT_STRIDE))

python3 "${SCRIPT_DIR}/qcar2_perception_server.py" \
  --stream-ip localhost \
  --stream-port "${DETECTION_PORT}" \
  --depth-port "${DEPTH_PORT}" \
  --qcar-id "${QCAR_ID}" \
  --virtual-port-stride "${VIRTUAL_PORT_STRIDE}" \
  --video3d-port "${VIDEO3D_PORT}" \
  --inference-mode "${INFERENCE_MODE}" &
PERCEPTION_PID=$!

cleanup() {
  if kill -0 "${PERCEPTION_PID}" >/dev/null 2>&1; then
    kill "${PERCEPTION_PID}" >/dev/null 2>&1 || true
    wait "${PERCEPTION_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

# Let the perception stream server start.
sleep 5

python3 "${SCRIPT_DIR}/qcar2_road_rules_driver.py" \
  --qcar-id "${QCAR_ID}" \
  --virtual-port-stride "${VIRTUAL_PORT_STRIDE}" \
  --detection-port "${DETECTION_PORT}" \
  --nodes "${NODES}" \
  --cruise-speed "${CRUISE_SPEED}" \
  --sample-rate "${SAMPLE_RATE}"
