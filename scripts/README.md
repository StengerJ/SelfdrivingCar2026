# QCar2 Single-File Autonomous Script

This folder uses a single control script:

- `qcar2_road_rules_driver.py`

It contains:

- lane keeping (based on Quanser lane-following example logic)
- ultralytics YOLO sign/object inference
- road-rule logic (stop sign, red light, pedestrian, yield, car obstacle)
- route following and direct QCar command output

## Run

```bash
bash scripts/run_competition_stack.sh
```

`run_competition_stack.sh` takes no arguments.
Edit defaults in that file if needed.

## Required model path

By default the launcher expects:

- `scripts/models/road_signs.pt`

If your model is somewhere else, edit `YOLO_MODEL_PATH` in `run_competition_stack.sh`.

## Notes

- Virtual QCar2 is forced automatically (no user prompt).
- No writing of `qcar_config.json` is required by this script path.
- Ensure `ultralytics` is installed in your container.
- Route generation is map-dynamic by default (reads the active map nodes/edges via HAL roadmap and builds a traversable node sequence automatically).
- Dynamic routing uses HAL roadmap APIs (`SDCSRoadMap`, `get_closest_node`, `find_shortest_path`, `generate_path`, `initial_check`) rather than custom graph traversal code.
- In `run_competition_stack.sh`, keep `NODES=""` for dynamic routing, or set `NODES="n1,n2,..."` to force a manual node route.
- Opposite-turn avoidance is enabled by default: for right-hand traffic it avoids left cross-over turns across every planned route leg (solid-yellow risk) and switches to alternate legal legs when available.
- A skyview printout is generated every run using HAL `roadmap.display()`, including plotted route waypoints and node order labels. Output goes to `scripts/skyview/`.
- Speed is now curve-adaptive by default with anti-surge control. Edit `run_competition_stack.sh` defaults (`CRUISE_SPEED`, `MIN_SPEED`, `MAX_THROTTLE`, `STEERING_K`, `MAP_STEER_WEIGHT`, `LANE_STEER_WEIGHT`, `CURVE_LAT_ACCEL`, `SPEED_UP_RATE`, `SPEED_DOWN_RATE`) to tune behavior.
- Steering is smoothed/rate-limited and lane blending is stronger to reduce sidewalk cutting and large recovery swings.
- Right-hand traffic uses right-side lane ROI for lane centering; dynamic route completion is gated by sequential node-arrival progress (not only waypoint index).
- Lane centering now includes yellow-line distance hold: tune `YELLOW_LINE_TARGET_RATIO` and `YELLOW_LINE_DISTANCE_GAIN` in `run_competition_stack.sh` to keep a fixed offset from the center yellow line.
- Yield behavior is slow-and-go (not full stop) unless conflicting hazards are detected.
- Crossing lane paint/center lines should not trigger halts; braking is gated by validated forward car detections (with stop-sign/red-light/pedestrian rules still active).
- To avoid post-run native-library segfaults in some containers, the script uses safe process exit by default; pass `--native-cleanup` only if you explicitly want `terminate()/close()` calls.
