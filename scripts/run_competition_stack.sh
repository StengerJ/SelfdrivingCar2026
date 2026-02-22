#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration. This launcher takes no command-line arguments.
QCAR_ID=0
VIRTUAL_PORT_STRIDE=10
NODES="10,2,4,14,20,22,9,7,0"
CRUISE_SPEED=0.55
SAMPLE_RATE=100
YOLO_MODEL_PATH="${SCRIPT_DIR}/models/road_signs.pt"
YOLO_CONFIDENCE=0.35
YOLO_IOU=0.45
YOLO_IMGSZ=640

if (($# > 0)); then
  echo "run_competition_stack.sh takes no arguments. Edit defaults inside this script if needed."
  exit 1
fi

if [[ ! -f "${YOLO_MODEL_PATH}" ]]; then
  echo "YOLO model not found at ${YOLO_MODEL_PATH}"
  echo "Set YOLO_MODEL_PATH in this script to your trained road-sign model."
  exit 1
fi

python3 "${SCRIPT_DIR}/qcar2_road_rules_driver.py" \
  --model-path "${YOLO_MODEL_PATH}" \
  --confidence "${YOLO_CONFIDENCE}" \
  --iou "${YOLO_IOU}" \
  --imgsz "${YOLO_IMGSZ}" \
  --qcar-id "${QCAR_ID}" \
  --virtual-port-stride "${VIRTUAL_PORT_STRIDE}" \
  --nodes "${NODES}" \
  --cruise-speed "${CRUISE_SPEED}" \
  --sample-rate "${SAMPLE_RATE}"

