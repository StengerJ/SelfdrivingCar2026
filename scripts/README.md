# QCar2 Single-File Autonomous Script

This folder now uses a single control script:

- `qcar2_road_rules_driver.py`

It contains:

- lane keeping (based on Quanser lane-following example logic)
- ultralytics YOLO sign/object inference
- road-rule logic (stop sign, red light, pedestrian, yield, car obstacle)
- route following and direct QCar command output

There is no server/client detection stream in the runtime path.

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

