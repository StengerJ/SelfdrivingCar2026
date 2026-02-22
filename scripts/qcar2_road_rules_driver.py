#!/usr/bin/env python3
"""Single-file QCar2 autonomy: lane keeping + YOLO road rules + route following."""

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from compat_qcar2_virtual import patch_qcar2_virtual_config

patch_qcar2_virtual_config()

from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from hal.utilities.control import PID, StanleyController
from hal.utilities.image_processing import ImageProcessing
from pal.products.qcar import IS_PHYSICAL_QCAR, QCar, QCarCameras, QCarGPS
from pal.utilities.math import Filter
from pit.YOLO.utils import QCar2DepthAligned


STOP_REQUESTED = False

BASE_HIL_PORT = 18960
BASE_VIDEO_PORT = 18961
BASE_GPS_PORT = 18967
BASE_LIDAR_IDEAL_PORT = 18968
BASE_VIDEO3D_PORT = 18965


class RuleState:
    def __init__(self):
        self.stop_state = "armed"
        self.stop_until = 0.0
        self.person_blocking = False


def parse_args():
    parser = argparse.ArgumentParser(description="QCar2 single-file autonomous driver")

    parser.add_argument("--model-path", default="", help="Path to ultralytics YOLO road-sign model")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--clipping-distance", type=float, default=10.0)

    parser.add_argument("--nodes", default="10,2,4,14,20,22,9,7,0")
    parser.add_argument("--left-hand-traffic", action="store_true")
    parser.add_argument("--small-map", action="store_true")
    parser.add_argument("--route-cyclic", action="store_true")

    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument("--runtime", type=float, default=900.0)
    parser.add_argument("--start-delay", type=float, default=1.0)
    parser.add_argument("--cruise-speed", type=float, default=0.55)
    parser.add_argument("--max-throttle", type=float, default=0.30)
    parser.add_argument("--max-steer", type=float, default=0.52)
    parser.add_argument("--stanley-gain", type=float, default=0.8)
    parser.add_argument("--speed-kp", type=float, default=0.18)
    parser.add_argument("--speed-ki", type=float, default=0.9)
    parser.add_argument("--steer-slow-gain", type=float, default=0.45)
    parser.add_argument("--min-corner-gain", type=float, default=0.45)

    parser.add_argument("--disable-lane-centering", action="store_true")
    parser.add_argument("--lane-cutoff-hz", type=float, default=25.0)
    parser.add_argument("--lane-steer-limit", type=float, default=0.5)
    parser.add_argument("--lane-steer-weight", type=float, default=1.0)
    parser.add_argument("--map-steer-weight", type=float, default=0.25)
    parser.add_argument("--camera-width", type=int, default=820)
    parser.add_argument("--camera-height", type=int, default=410)
    parser.add_argument("--camera-fps", type=float, default=30.0)

    parser.add_argument("--stop-sign-trigger", type=float, default=0.65)
    parser.add_argument("--stop-hold", type=float, default=2.5)
    parser.add_argument("--stop-cooldown", type=float, default=6.0)
    parser.add_argument("--person-stop", type=float, default=1.25)
    parser.add_argument("--person-resume", type=float, default=1.45)
    parser.add_argument("--traffic-light-stop", type=float, default=1.70)
    parser.add_argument("--yield-trigger", type=float, default=1.00)
    parser.add_argument("--yield-gain", type=float, default=0.50)
    parser.add_argument("--car-stop", type=float, default=0.40)
    parser.add_argument("--car-slow", type=float, default=1.20)
    parser.add_argument("--forward-obstacle-max", type=float, default=1.2)

    parser.add_argument("--calibrate-gps", action="store_true")
    parser.add_argument("--calibration-pose", default="0,2,-1.5708")
    parser.add_argument("--print-rate", type=float, default=2.0)

    parser.add_argument("--qcar-id", type=int, default=0)
    parser.add_argument("--virtual-port-stride", type=int, default=10)
    parser.add_argument("--hil-port", type=int, default=None)
    parser.add_argument("--video-port", type=int, default=None)
    parser.add_argument("--gps-port", type=int, default=None)
    parser.add_argument("--lidar-ideal-port", type=int, default=None)
    parser.add_argument("--video3d-port", type=int, default=None)
    parser.add_argument("--depth-port", type=str, default="18777")
    parser.add_argument("--qlabs-host", default="localhost")
    parser.add_argument(
        "--qlabs-camera",
        choices=["none", "trailing", "overhead", "front", "right", "back", "left", "rgb", "depth"],
        default="trailing",
    )
    parser.add_argument("--skip-qlabs-attach", action="store_true")
    parser.add_argument("--qlabs-setup", action="store_true")

    return parser.parse_args()


def parse_nodes(nodes_str):
    return [int(token.strip()) for token in nodes_str.split(",") if token.strip()]


def parse_pose(pose_str):
    values = [float(token.strip()) for token in pose_str.split(",") if token.strip()]
    if len(values) != 3:
        raise ValueError("calibration pose must contain exactly 3 comma-separated values")
    return values


def signal_handler(*_args):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def resolve_virtual_port(explicit_port, base_port, qcar_id, stride):
    if explicit_port is not None:
        return int(explicit_port)
    return int(base_port + qcar_id * stride)


def maybe_run_qlabs_setup(args):
    if IS_PHYSICAL_QCAR or not args.qlabs_setup:
        return
    if args.qcar_id != 0:
        print("Warning: --qlabs-setup spawns actor 0 only. Skipping setup because --qcar-id is not 0.")
        return

    virtual_dir = Path(__file__).resolve().parents[3] / "python_resources" / "qcar2" / "virtual"
    if virtual_dir.exists():
        sys.path.insert(0, str(virtual_dir))
    try:
        from qlabs_setup_applications import setup as qlabs_setup
    except Exception as exc:
        print(f"Warning: could not import qlabs_setup_applications.py: {exc}")
        return

    print("Launching QLabs QCar2 virtual setup...")
    try:
        qlabs_setup()
    except Exception as exc:
        print(f"Warning: QLabs setup failed: {exc}")


def maybe_attach_qlabs_actor(args):
    if IS_PHYSICAL_QCAR or args.skip_qlabs_attach:
        return None

    try:
        from qvl.qlabs import QuanserInteractiveLabs
        from qvl.qcar2 import QLabsQCar2
    except Exception as exc:
        print(f"Warning: qvl package unavailable, skipping QLabs attach: {exc}")
        return None

    qlabs = QuanserInteractiveLabs()
    if not qlabs.open(args.qlabs_host):
        print(f"Warning: could not connect to QLabs at {args.qlabs_host}, skipping actor attach.")
        return None

    actor = QLabsQCar2(qlabs)
    actor.actorNumber = args.qcar_id
    camera_map = {
        "trailing": actor.CAMERA_TRAILING,
        "overhead": actor.CAMERA_OVERHEAD,
        "front": actor.CAMERA_CSI_FRONT,
        "right": actor.CAMERA_CSI_RIGHT,
        "back": actor.CAMERA_CSI_BACK,
        "left": actor.CAMERA_CSI_LEFT,
        "rgb": actor.CAMERA_RGB,
        "depth": actor.CAMERA_DEPTH,
    }
    if args.qlabs_camera != "none":
        try:
            actor.possess(camera_map[args.qlabs_camera])
        except Exception as exc:
            print(f"Warning: QLabs possess failed for actor {args.qcar_id}: {exc}")
    return qlabs


def configure_virtual_depth_camera(depth_rgb, video3d_port):
    if getattr(depth_rgb, "isPhysical", True):
        return
    if video3d_port == BASE_VIDEO3D_PORT:
        return
    try:
        from pal.utilities.vision import Camera3D

        if hasattr(depth_rgb, "camera") and depth_rgb.camera is not None:
            try:
                depth_rgb.camera.terminate()
            except Exception:
                pass
        depth_rgb.camera = Camera3D(
            mode="RGB, Depth",
            deviceId=f"0@tcpip://localhost:{video3d_port}",
            frameWidthRGB=640,
            frameHeightRGB=480,
            frameRateRGB=30,
            frameWidthDepth=640,
            frameHeightDepth=480,
            frameRateDepth=15,
            frameWidthIR=640,
            frameHeightIR=480,
            frameRateIR=30,
        )
    except Exception as exc:
        print(f"Warning: failed to bind virtual RealSense to port {video3d_port}: {exc}")


def compute_leds(throttle, steering):
    leds = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)
    if steering > 0.18:
        leds[0] = 1
        leds[2] = 1
    elif steering < -0.18:
        leds[1] = 1
        leds[3] = 1
    if throttle < -0.02:
        leds[5] = 1
    return leds


def compute_lane_steering_example(front_bgr, dt, steering_filter, lane_steer_limit):
    h, w = front_bgr.shape[:2]
    row_start = int(np.clip(round((524.0 / 820.0) * h), 0, h - 1))
    row_end = int(np.clip(round((674.0 / 820.0) * h), row_start + 1, h))
    col_end = int(np.clip(round((820.0 / 1640.0) * w), 1, w))
    cropped = front_bgr[row_start:row_end, 0:col_end]
    if cropped.size == 0:
        return None

    hsv_buf = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    binary = ImageProcessing.binary_thresholding(
        frame=hsv_buf,
        lowerBounds=np.array([10, 50, 100], dtype=np.uint8),
        upperBounds=np.array([45, 255, 255], dtype=np.uint8),
    )
    slope, intercept = ImageProcessing.find_slope_intercept_from_binary(binary=binary)
    raw_steering = 1.5 * (slope - 0.3419) + (1.0 / 150.0) * (intercept + 5.0)
    clipped = float(np.clip(raw_steering, -lane_steer_limit, lane_steer_limit))
    try:
        steering = float(steering_filter.send((clipped, dt)))
    except Exception:
        steering = clipped
    if np.isnan(steering):
        return None
    return steering


def depth_plane(depth):
    if depth.ndim == 3 and depth.shape[2] == 1:
        return depth[:, :, 0]
    return depth


def median_depth_from_bbox(depth, bbox_xyxy, inset=0.2):
    d = depth_plane(depth)
    x1, y1, x2, y2 = bbox_xyxy
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    xx1 = max(0, int(x1 + w * inset))
    yy1 = max(0, int(y1 + h * inset))
    xx2 = min(d.shape[1], int(x2 - w * inset))
    yy2 = min(d.shape[0], int(y2 - h * inset))
    if xx2 <= xx1 or yy2 <= yy1:
        return float("inf")
    roi = d[yy1:yy2, xx1:xx2]
    valid = roi[np.isfinite(roi)]
    valid = valid[(valid > 0) & (valid < 100)]
    if valid.size == 0:
        return float("inf")
    return float(np.median(valid))


def forward_obstacle_distance(depth):
    d = depth_plane(depth)
    h, w = d.shape[:2]
    y1 = int(0.45 * h)
    y2 = int(0.90 * h)
    x1 = int(0.35 * w)
    x2 = int(0.65 * w)
    roi = d[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi)]
    valid = valid[(valid > 0) & (valid < 100)]
    if valid.size == 0:
        return float("inf")
    return float(np.percentile(valid, 25))


def classify_name(name):
    n = str(name).lower().replace("_", " ").strip()
    if ("stop" in n and "sign" in n) or n == "stop":
        return "stop"
    if "yield" in n:
        return "yield"
    if "traffic" in n and "light" in n:
        return "traffic_red"
    if "traffic" in n and "red" in n:
        return "traffic_red"
    if "red light" in n:
        return "traffic_red"
    if "person" in n or "pedestrian" in n:
        return "person"
    if "car" in n or "vehicle" in n:
        return "car"
    return None


def class_name_from_index(names, idx):
    if isinstance(names, dict):
        return str(names.get(idx, idx))
    if isinstance(names, (list, tuple)) and 0 <= idx < len(names):
        return str(names[idx])
    return str(idx)


def detect_distances(model, rgb, depth, args):
    distances = {
        "stop": float("inf"),
        "person": float("inf"),
        "car": float("inf"),
        "yield": float("inf"),
        "traffic_red": float("inf"),
    }

    results = model.predict(
        source=rgb,
        conf=args.confidence,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False,
        device=args.device or None,
        half=False,
    )
    result = results[0]
    names = result.names

    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy().astype(np.int32)
        cls_idx = result.boxes.cls.cpu().numpy().astype(np.int32)
        for i in range(len(xyxy)):
            mapped = classify_name(class_name_from_index(names, int(cls_idx[i])))
            if mapped is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy[i].tolist()]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(rgb.shape[1], x2)
            y2 = min(rgb.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            d = median_depth_from_bbox(depth, (x1, y1, x2, y2), inset=0.2)
            if not np.isfinite(d) or d <= 0 or d > args.clipping_distance:
                continue
            distances[mapped] = min(distances[mapped], d)

    obs_d = forward_obstacle_distance(depth)
    if obs_d < args.forward_obstacle_max:
        distances["car"] = min(distances["car"], obs_d)

    return distances


def compute_rule_gain(now, dists, state, args):
    if state.person_blocking:
        state.person_blocking = dists["person"] < args.person_resume
    else:
        state.person_blocking = dists["person"] < args.person_stop

    if state.stop_state == "cooldown" and now >= state.stop_until:
        state.stop_state = "armed"
    if state.stop_state == "armed" and dists["stop"] < args.stop_sign_trigger:
        state.stop_state = "stopping"
        state.stop_until = now + args.stop_hold
    if state.stop_state == "stopping" and now >= state.stop_until:
        state.stop_state = "cooldown"
        state.stop_until = now + args.stop_cooldown

    gain = 1.0
    if state.person_blocking or state.stop_state == "stopping":
        gain = 0.0
    if dists["traffic_red"] < args.traffic_light_stop:
        gain = 0.0

    if dists["car"] < args.car_slow:
        if args.car_slow <= args.car_stop:
            gain *= 0.0
        else:
            car_gain = np.clip((dists["car"] - args.car_stop) / (args.car_slow - args.car_stop), 0.0, 1.0)
            gain *= float(car_gain)
    if dists["yield"] < args.yield_trigger:
        gain *= float(np.clip(args.yield_gain, 0.0, 1.0))

    return float(np.clip(gain, 0.0, 1.0))


def main():
    global STOP_REQUESTED
    args = parse_args()
    signal.signal(signal.SIGINT, signal_handler)

    if args.qcar_id < 0 or args.virtual_port_stride < 0:
        raise ValueError("qcar-id and virtual-port-stride must be >= 0")
    if not args.model_path:
        raise RuntimeError("Missing --model-path")
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is not installed in this environment") from exc

    maybe_run_qlabs_setup(args)

    hil_port = resolve_virtual_port(args.hil_port, BASE_HIL_PORT, args.qcar_id, args.virtual_port_stride)
    video_port = resolve_virtual_port(args.video_port, BASE_VIDEO_PORT, args.qcar_id, args.virtual_port_stride)
    gps_port = resolve_virtual_port(args.gps_port, BASE_GPS_PORT, args.qcar_id, args.virtual_port_stride)
    lidar_ideal_port = resolve_virtual_port(args.lidar_ideal_port, BASE_LIDAR_IDEAL_PORT, args.qcar_id, args.virtual_port_stride)
    video3d_port = resolve_virtual_port(args.video3d_port, BASE_VIDEO3D_PORT, args.qcar_id, args.virtual_port_stride)

    node_sequence = parse_nodes(args.nodes)
    calibration_pose = parse_pose(args.calibration_pose)

    roadmap = SDCSRoadMap(leftHandTraffic=args.left_hand_traffic, useSmallMap=args.small_map)
    try:
        waypoint_sequence = roadmap.generate_path(node_sequence)
    except TypeError:
        waypoint_sequence = roadmap.generate_path(nodeSequence=node_sequence)
    if waypoint_sequence is None:
        raise RuntimeError("Could not generate a route for the provided node sequence.")

    initial_pose = roadmap.get_node_pose(node_sequence[0]).squeeze()
    steering_controller = StanleyController(waypoints=waypoint_sequence, k=args.stanley_gain, cyclic=args.route_cyclic)
    steering_controller.maxSteeringAngle = args.max_steer
    speed_controller = PID(Kp=args.speed_kp, Ki=args.speed_ki, Kd=0.0, uLimits=(-args.max_throttle, args.max_throttle))
    lane_filter = Filter().low_pass_first_order_variable(args.lane_cutoff_hz, 0.033)
    next(lane_filter)

    qlabs_client = maybe_attach_qlabs_actor(args)
    model = YOLO(args.model_path)
    rule_state = RuleState()

    qcar = QCar(readMode=1, frequency=int(args.sample_rate), hilPort=hil_port)
    gps = QCarGPS(initialPose=calibration_pose, calibrate=args.calibrate_gps, gpsPort=gps_port, lidarIdealPort=lidar_ideal_port)
    cameras = QCarCameras(
        frameWidth=args.camera_width,
        frameHeight=args.camera_height,
        frameRate=args.camera_fps,
        enableFront=True,
        videoPort=video_port,
    )
    depth_rgb = QCar2DepthAligned(port=args.depth_port)
    configure_virtual_depth_camera(depth_rgb, video3d_port)
    ekf = QCarEKF(x_0=initial_pose)

    print("QCar2 single-file autonomy started")
    print("Route nodes:", node_sequence)
    print("Running on physical QCar:", IS_PHYSICAL_QCAR)
    print("QLabs actor ID:", args.qcar_id)
    print("Model:", args.model_path)

    start_node_reached = True
    init_steering_controller = None

    try:
        t_wait = time.time()
        while not STOP_REQUESTED and time.time() - t_wait < 8.0:
            if gps.readGPS():
                init_pose = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                start_node_reached, init_waypoints = roadmap.initial_check(init_pose, node_sequence, waypoint_sequence)
                if not start_node_reached and init_waypoints is not None:
                    init_steering_controller = StanleyController(waypoints=init_waypoints, k=args.stanley_gain, cyclic=False)
                    init_steering_controller.maxSteeringAngle = args.max_steer
                break
            time.sleep(0.02)

        dt_target = 1.0 / args.sample_rate
        t0 = time.time()
        t_prev = t0
        next_print = t0
        delta = 0.0

        while not STOP_REQUESTED and (time.time() - t0) < args.runtime:
            loop_start = time.time()
            dt = max(loop_start - t_prev, 1e-3)
            t_prev = loop_start

            qcar.read()
            cameras.readAll()
            depth_rgb.read()

            if gps.readGPS():
                y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
            else:
                ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])

            x = float(ekf.x_hat[0, 0])
            y = float(ekf.x_hat[1, 0])
            th = float(ekf.x_hat[2, 0])
            p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            v = float(qcar.motorTach)

            if loop_start - t0 < args.start_delay:
                throttle = 0.0
                delta = 0.0
                dists = {"stop": np.inf, "person": np.inf, "car": np.inf, "yield": np.inf, "traffic_red": np.inf}
                rule_gain = 1.0
            else:
                v_for_steering = max(abs(v), 0.05)
                if not start_node_reached and init_steering_controller is not None:
                    dist_to_start = np.linalg.norm(waypoint_sequence[:, 0] - p)
                    start_node_reached = dist_to_start < 0.2
                    if start_node_reached:
                        delta_map = steering_controller.update(p, th, v_for_steering)
                    else:
                        delta_map = init_steering_controller.update(p, th, v_for_steering)
                else:
                    delta_map = steering_controller.update(p, th, v_for_steering)

                lane_steer = None
                if not args.disable_lane_centering and cameras.csiFront is not None:
                    lane_steer = compute_lane_steering_example(
                        front_bgr=cameras.csiFront.imageData,
                        dt=dt,
                        steering_filter=lane_filter,
                        lane_steer_limit=args.lane_steer_limit,
                    )
                if lane_steer is None:
                    delta = float(np.clip(delta_map, -args.max_steer, args.max_steer))
                else:
                    delta = float(np.clip(args.map_steer_weight * delta_map + args.lane_steer_weight * lane_steer, -args.max_steer, args.max_steer))

                dists = detect_distances(model, depth_rgb.rgb, depth_rgb.depth, args)
                rule_gain = compute_rule_gain(loop_start, dists, rule_state, args)

                corner_gain = np.clip(
                    1.0 - args.steer_slow_gain * abs(delta) / max(args.max_steer, 1e-3),
                    args.min_corner_gain,
                    1.0,
                )
                v_ref = args.cruise_speed * corner_gain * rule_gain

                if steering_controller.pathComplete and not args.route_cyclic:
                    v_ref = 0.0

                throttle = float(speed_controller.update(v_ref, v, dt))
                if v_ref <= 0.01 and v > 0.05:
                    throttle = min(throttle, -0.08)
                elif v_ref <= 0.01:
                    throttle = 0.0

                if args.print_rate > 0 and loop_start >= next_print:
                    next_print = loop_start + (1.0 / args.print_rate)
                    print(
                        f"t={loop_start - t0:6.1f}s "
                        f"v={v:4.2f}m/s v_ref={v_ref:4.2f} "
                        f"thr={throttle:5.2f} str={delta:5.2f} "
                        f"stop={dists['stop']:.2f} ped={dists['person']:.2f} "
                        f"car={dists['car']:.2f} red={dists['traffic_red']:.2f} "
                        f"yield={dists['yield']:.2f} gain={rule_gain:.2f}"
                    )

            leds = compute_leds(throttle, delta)
            qcar.read_write_std(throttle=throttle, steering=delta, LEDs=leds)

            if steering_controller.pathComplete and not args.route_cyclic and abs(v) < 0.03:
                print("Route complete and vehicle stopped.")
                break

            elapsed = time.time() - loop_start
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)

    finally:
        try:
            qcar.read_write_std(throttle=0.0, steering=0.0, LEDs=np.zeros(8, dtype=np.float64))
        except Exception:
            pass
        depth_rgb.terminate()
        cameras.terminate()
        gps.terminate()
        qcar.terminate()
        if qlabs_client is not None:
            try:
                qlabs_client.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
