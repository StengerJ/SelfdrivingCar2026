#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration. This launcher takes no command-line arguments.
QCAR_ID=0
VIRTUAL_PORT_STRIDE=10
# Leave blank to auto-generate route from current map graph.
NODES=""
AUTO_ROUTE_MAX_NODES=0
AUTO_ROUTE_START_NODE=-1
OPPOSITE_TURN_THRESHOLD_DEG=18
CRUISE_SPEED=0.28
MIN_SPEED=0.09
MAX_THROTTLE=0.11
MAX_STEER=0.46
STEER_RATE_LIMIT=0.65
STEERING_K=0.55
NODE_ARRIVAL_RADIUS=0.30
WAYPOINT_SWITCH_DISTANCE=0.35
MAP_STEER_WEIGHT=0.95
LANE_STEER_WEIGHT=0.45
YELLOW_LINE_TARGET_RATIO=0.30
YELLOW_LINE_DISTANCE_GAIN=0.45
YELLOW_LINE_BOTTOM_ROI=0.55
LANE_TURN_DISABLE_RATIO=0.90
CURVE_LAT_ACCEL=0.28
SPEED_UP_RATE=0.20
SPEED_DOWN_RATE=2.40
YIELD_GAIN=0.90
YIELD_CLEAR_CAR_DISTANCE=0.85
YIELD_CLEAR_OBSTACLE_DISTANCE=0.50
CAR_FORWARD_X_MIN=0.28
CAR_FORWARD_X_MAX=0.72
CAR_FORWARD_Y_MIN=0.30
CAR_MIN_AREA_RATIO=0.0015
OBSTACLE_STOP=0.22
OBSTACLE_SLOW=0.75
SAMPLE_RATE=100
SKYVIEW_OUTPUT_DIR="${SCRIPT_DIR}/skyview"
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

CMD=(
python3 "${SCRIPT_DIR}/qcar2_road_rules_driver.py"
  --model-path "${YOLO_MODEL_PATH}"
  --confidence "${YOLO_CONFIDENCE}"
  --iou "${YOLO_IOU}"
  --imgsz "${YOLO_IMGSZ}"
  --qcar-id "${QCAR_ID}"
  --virtual-port-stride "${VIRTUAL_PORT_STRIDE}"
  --auto-route-max-nodes "${AUTO_ROUTE_MAX_NODES}"
  --auto-route-start-node "${AUTO_ROUTE_START_NODE}"
  --opposite-turn-threshold-deg "${OPPOSITE_TURN_THRESHOLD_DEG}"
  --cruise-speed "${CRUISE_SPEED}"
  --min-speed "${MIN_SPEED}"
  --max-throttle "${MAX_THROTTLE}"
  --max-steer "${MAX_STEER}"
  --steer-rate-limit "${STEER_RATE_LIMIT}"
  --steering-k "${STEERING_K}"
  --node-arrival-radius "${NODE_ARRIVAL_RADIUS}"
  --waypoint-switch-distance "${WAYPOINT_SWITCH_DISTANCE}"
  --map-steer-weight "${MAP_STEER_WEIGHT}"
  --lane-steer-weight "${LANE_STEER_WEIGHT}"
  --yellow-line-target-ratio "${YELLOW_LINE_TARGET_RATIO}"
  --yellow-line-distance-gain "${YELLOW_LINE_DISTANCE_GAIN}"
  --yellow-line-bottom-roi "${YELLOW_LINE_BOTTOM_ROI}"
  --lane-turn-disable-ratio "${LANE_TURN_DISABLE_RATIO}"
  --curve-lat-accel "${CURVE_LAT_ACCEL}"
  --speed-up-rate "${SPEED_UP_RATE}"
  --speed-down-rate "${SPEED_DOWN_RATE}"
  --yield-gain "${YIELD_GAIN}"
  --yield-clear-car-distance "${YIELD_CLEAR_CAR_DISTANCE}"
  --yield-clear-obstacle-distance "${YIELD_CLEAR_OBSTACLE_DISTANCE}"
  --car-forward-x-min "${CAR_FORWARD_X_MIN}"
  --car-forward-x-max "${CAR_FORWARD_X_MAX}"
  --car-forward-y-min "${CAR_FORWARD_Y_MIN}"
  --car-min-area-ratio "${CAR_MIN_AREA_RATIO}"
  --obstacle-stop "${OBSTACLE_STOP}"
  --obstacle-slow "${OBSTACLE_SLOW}"
  --skyview-output-dir "${SKYVIEW_OUTPUT_DIR}"
  --sample-rate "${SAMPLE_RATE}"
)

if [[ -n "${NODES}" ]]; then
  CMD+=(--nodes "${NODES}")
fi

"${CMD[@]}"
