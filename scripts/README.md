# QCar2 Competition Scripts

## QLabs Digital Twin Notes

- These scripts are configured for the virtual QCar2 in QLabs (not physical hardware).
- Use the QCar actor ID from your test script (default in provided examples is `0`).
- Virtual ports are derived as `base + qcar_id * 10` unless you override them manually.
- Default perception backend is custom OpenCV inference (not YOLO) to avoid YOLO runtime issues.

## 1) Start perception stream (stop signs + pedestrians + traffic objects)

```bash
python3 scripts/qcar2_perception_server.py \
  --qcar-id 0 \
  --stream-ip localhost \
  --stream-port 18666 \
  --depth-port 18777 \
  --inference-mode custom
```

## 2) Start autonomous driver (nodes + lane centering + road rules)

```bash
python3 scripts/qcar2_road_rules_driver.py \
  --qcar-id 0 \
  --nodes "10,2,4,14,20,22,9,7,0" \
  --cruise-speed 0.65 \
  --sample-rate 100
```

## 3) One-command launch (recommended, no arguments)

```bash
bash scripts/run_competition_stack.sh
```

`run_competition_stack.sh` now takes no command-line arguments.
Edit defaults directly in the script if you want a different route/speed/ID.

## Notes

- Keep `qcar2_perception_server.py` running while `qcar2_road_rules_driver.py` is active.
- The driver stops for:
  - stop signs (`--stop-sign-trigger`, `--stop-hold`, `--stop-cooldown`)
  - pedestrians (`--person-stop`, `--person-resume`)
- Lane centering can be disabled with `--disable-lane-centering`.
- YOLO rule gating can be disabled with `--disable-yolo-rules`.
