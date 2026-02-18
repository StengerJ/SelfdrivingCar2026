#!/usr/bin/env python3
"""Route-following controller for QCar2 with lane centering and road-rule stops."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from compat_qcar2_virtual import patch_qcar2_virtual_config

patch_qcar2_virtual_config()

from competition_common import (
    ROW_PERSON,
    ROW_STOP_SIGN,
    DetectionStreamClient,
    RoadRuleConfig,
    RoadRuleStateMachine,
)
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from hal.utilities.control import PID, StanleyController
from hal.utilities.image_processing import ImageProcessing
from pal.products.qcar import IS_PHYSICAL_QCAR, QCar, QCarCameras, QCarGPS


STOP_REQUESTED = False

BASE_HIL_PORT = 18960
BASE_VIDEO_PORT = 18961
BASE_GPS_PORT = 18967
BASE_LIDAR_IDEAL_PORT = 18968


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QCar2 competition driver: node routing + lane centering + stop sign/pedestrian compliance"
    )
    parser.add_argument("--nodes", default="10,2,4,14,20,22,9,7,0", help="Comma-separated SDCS node sequence")
    parser.add_argument("--left-hand-traffic", action="store_true", help="Use left-hand traffic map variant")
    parser.add_argument("--small-map", action="store_true", help="Use small SDCS map")
    parser.add_argument("--route-cyclic", action="store_true", help="Loop route continuously")
    parser.add_argument(
        "--qcar-id",
        type=int,
        default=0,
        help="QLabs QCar2 actor number. Test scripts use actor 0 by default.",
    )
    parser.add_argument(
        "--virtual-port-stride",
        type=int,
        default=10,
        help="Port offset per QCar ID in virtual mode. Effective port = base + qcar_id*stride.",
    )
    parser.add_argument("--hil-port", type=int, default=None, help="Override QCar HIL port (virtual mode)")
    parser.add_argument("--video-port", type=int, default=None, help="Override CSI base video port (virtual mode)")
    parser.add_argument("--gps-port", type=int, default=None, help="Override GPS stream port")
    parser.add_argument("--lidar-ideal-port", type=int, default=None, help="Override lidar-ideal stream port")
    parser.add_argument("--qlabs-host", default="localhost", help="QLabs host for actor attach/possess")
    parser.add_argument(
        "--qlabs-camera",
        choices=["none", "trailing", "overhead", "front", "right", "back", "left", "rgb", "depth"],
        default="trailing",
        help="QLabs camera to possess for the selected actor",
    )
    parser.add_argument("--skip-qlabs-attach", action="store_true", help="Skip QLabs actor attach/possess step")
    parser.add_argument(
        "--qlabs-setup",
        action="store_true",
        help="Reset and spawn the default QCar2 virtual scene before starting",
    )

    parser.add_argument("--sample-rate", type=float, default=100.0, help="Control loop rate (Hz)")
    parser.add_argument("--runtime", type=float, default=900.0, help="Maximum run time (s)")
    parser.add_argument("--start-delay", type=float, default=1.0, help="Delay before controller engages (s)")
    parser.add_argument("--cruise-speed", type=float, default=0.65, help="Nominal speed (m/s)")
    parser.add_argument("--max-throttle", type=float, default=0.30, help="Throttle saturation")
    parser.add_argument("--max-steer", type=float, default=0.52, help="Steering saturation (rad)")
    parser.add_argument("--stanley-gain", type=float, default=0.8, help="Stanley cross-track gain")
    parser.add_argument("--speed-kp", type=float, default=0.18, help="Speed PID Kp")
    parser.add_argument("--speed-ki", type=float, default=0.9, help="Speed PID Ki")
    parser.add_argument("--steer-slow-gain", type=float, default=0.45, help="Speed reduction vs steering demand")
    parser.add_argument("--min-corner-gain", type=float, default=0.45, help="Minimum speed multiplier while cornering")

    parser.add_argument("--disable-lane-centering", action="store_true", help="Disable camera-based lane centering")
    parser.add_argument("--lane-gain", type=float, default=0.18, help="Lane-centering correction gain")
    parser.add_argument("--lane-max-correction", type=float, default=0.20, help="Max absolute lane correction (rad)")
    parser.add_argument("--camera-width", type=int, default=820, help="Front CSI width")
    parser.add_argument("--camera-height", type=int, default=410, help="Front CSI height")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="Front CSI frame rate")

    parser.add_argument("--disable-yolo-rules", action="store_true", help="Disable stop-sign/person rule gating")
    parser.add_argument("--detection-ip", default="localhost", help="Detection stream host")
    parser.add_argument("--detection-port", type=int, default=18666, help="Detection stream port")
    parser.add_argument("--stop-sign-trigger", type=float, default=0.65, help="Stop sign trigger distance (m)")
    parser.add_argument("--stop-hold", type=float, default=2.5, help="Stop sign full-stop hold time (s)")
    parser.add_argument("--stop-cooldown", type=float, default=6.0, help="Stop sign re-trigger cooldown (s)")
    parser.add_argument("--person-stop", type=float, default=1.25, help="Pedestrian stop distance (m)")
    parser.add_argument("--person-resume", type=float, default=1.45, help="Pedestrian resume distance (m)")

    parser.add_argument("--calibrate-gps", action="store_true", help="Run QCarGPS calibration routine on startup")
    parser.add_argument(
        "--calibration-pose",
        default="0,2,-1.5708",
        help="Initial GPS calibration pose: x,y,heading_rad",
    )
    parser.add_argument("--print-rate", type=float, default=2.0, help="Status print rate (Hz)")
    return parser.parse_args()


def resolve_virtual_port(explicit_port: int | None, base_port: int, qcar_id: int, stride: int) -> int:
    if explicit_port is not None:
        return int(explicit_port)
    return int(base_port + qcar_id * stride)


def maybe_run_qlabs_setup(args: argparse.Namespace) -> None:
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


def maybe_attach_qlabs_actor(args: argparse.Namespace):
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


def parse_nodes(nodes_str: str) -> list[int]:
    return [int(token.strip()) for token in nodes_str.split(",") if token.strip()]


def parse_pose(pose_str: str) -> list[float]:
    values = [float(token.strip()) for token in pose_str.split(",") if token.strip()]
    if len(values) != 3:
        raise ValueError("calibration pose must contain exactly 3 comma-separated values")
    return values


def signal_handler(*_args) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def compute_leds(throttle: float, steering: float) -> np.ndarray:
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


def compute_lane_error(front_bgr: np.ndarray) -> float | None:
    """Estimate lane-center offset in normalized image coordinates."""
    h, w = front_bgr.shape[:2]
    roi = front_bgr[int(0.55 * h) :, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Yellow + white lane mask.
    yellow = ImageProcessing.binary_thresholding(
        frame=hsv,
        lowerBounds=np.array([10, 40, 90], dtype=np.uint8),
        upperBounds=np.array([45, 255, 255], dtype=np.uint8),
    )
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    white = ImageProcessing.binary_thresholding(
        frame=gray,
        lowerBounds=170,
        upperBounds=255,
    ).astype(np.uint8)
    lane_mask = cv2.bitwise_or(yellow, white)
    lane_mask = ImageProcessing.image_filtering_open(
        lane_mask,
        dilate=1,
        erode=1,
        total=1,
    )

    # Use a lower band where lane boundaries are strongest.
    band = lane_mask[int(0.60 * lane_mask.shape[0]) :, :]
    histogram = np.sum(band, axis=0)
    midpoint = w // 2

    left_hist = histogram[:midpoint]
    right_hist = histogram[midpoint:]

    left_peak = int(np.argmax(left_hist)) if left_hist.max() > 1000 else None
    right_peak = int(np.argmax(right_hist) + midpoint) if right_hist.max() > 1000 else None

    if left_peak is None and right_peak is None:
        return None

    lane_width_guess = int(0.42 * w)
    if left_peak is None:
        lane_center = right_peak - lane_width_guess / 2.0
    elif right_peak is None:
        lane_center = left_peak + lane_width_guess / 2.0
    else:
        lane_center = 0.5 * (left_peak + right_peak)

    return float((lane_center - (w / 2.0)) / (w / 2.0))


def main() -> None:
    global STOP_REQUESTED
    args = parse_args()
    if args.qcar_id < 0:
        raise ValueError("--qcar-id must be >= 0")
    if args.virtual_port_stride < 0:
        raise ValueError("--virtual-port-stride must be >= 0")
    signal.signal(signal.SIGINT, signal_handler)

    maybe_run_qlabs_setup(args)

    hil_port = resolve_virtual_port(args.hil_port, BASE_HIL_PORT, args.qcar_id, args.virtual_port_stride)
    video_port = resolve_virtual_port(args.video_port, BASE_VIDEO_PORT, args.qcar_id, args.virtual_port_stride)
    gps_port = resolve_virtual_port(args.gps_port, BASE_GPS_PORT, args.qcar_id, args.virtual_port_stride)
    lidar_ideal_port = resolve_virtual_port(
        args.lidar_ideal_port,
        BASE_LIDAR_IDEAL_PORT,
        args.qcar_id,
        args.virtual_port_stride,
    )

    node_sequence = parse_nodes(args.nodes)
    calibration_pose = parse_pose(args.calibration_pose)

    roadmap = SDCSRoadMap(
        leftHandTraffic=args.left_hand_traffic,
        useSmallMap=args.small_map,
    )
    # Support both older/newer HAL API variants for SDCSRoadMap.generate_path.
    try:
        waypoint_sequence = roadmap.generate_path(node_sequence)
    except TypeError:
        waypoint_sequence = roadmap.generate_path(nodeSequence=node_sequence)
    if waypoint_sequence is None:
        raise RuntimeError("Could not generate a route for the provided node sequence.")

    initial_pose = roadmap.get_node_pose(node_sequence[0]).squeeze()
    steering_controller = StanleyController(
        waypoints=waypoint_sequence,
        k=args.stanley_gain,
        cyclic=args.route_cyclic,
    )
    steering_controller.maxSteeringAngle = args.max_steer

    speed_controller = PID(
        Kp=args.speed_kp,
        Ki=args.speed_ki,
        Kd=0.0,
        uLimits=(-args.max_throttle, args.max_throttle),
    )

    rule_machine = RoadRuleStateMachine(
        RoadRuleConfig(
            stop_sign_trigger_m=args.stop_sign_trigger,
            stop_sign_hold_s=args.stop_hold,
            stop_sign_cooldown_s=args.stop_cooldown,
            person_stop_m=args.person_stop,
            person_resume_m=args.person_resume,
        )
    )

    detection_client = None
    if not args.disable_yolo_rules:
        detection_client = DetectionStreamClient(
            ip=args.detection_ip,
            port=args.detection_port,
            non_blocking=True,
        )

    qlabs_client = maybe_attach_qlabs_actor(args)

    qcar = QCar(readMode=1, frequency=int(args.sample_rate), hilPort=hil_port)
    gps = QCarGPS(
        initialPose=calibration_pose,
        calibrate=args.calibrate_gps,
        gpsPort=gps_port,
        lidarIdealPort=lidar_ideal_port,
    )
    cameras = QCarCameras(
        frameWidth=args.camera_width,
        frameHeight=args.camera_height,
        frameRate=args.camera_fps,
        enableFront=True,
        videoPort=video_port,
    )
    ekf = QCarEKF(x_0=initial_pose)

    print("Route nodes:", node_sequence)
    print("Running on physical QCar:", IS_PHYSICAL_QCAR)
    print("QLabs actor ID:", args.qcar_id)
    print(
        "I/O ports:",
        {
            "hil": hil_port,
            "video": video_port,
            "gps": gps_port,
            "lidar_ideal": lidar_ideal_port,
        },
    )
    print("YOLO road-rule gating enabled:", detection_client is not None)

    start_node_reached = True
    init_steering_controller = None

    try:
        # Establish an initial GPS estimate.
        t_wait = time.time()
        while not STOP_REQUESTED and time.time() - t_wait < 8.0:
            if gps.readGPS():
                init_pose = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                start_node_reached, init_waypoints = roadmap.initial_check(
                    init_pose,
                    node_sequence,
                    waypoint_sequence,
                )
                if not start_node_reached and init_waypoints is not None:
                    init_steering_controller = StanleyController(
                        waypoints=init_waypoints,
                        k=args.stanley_gain,
                        cyclic=False,
                    )
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

                lane_delta = 0.0
                if not args.disable_lane_centering and cameras.csiFront is not None:
                    lane_error = compute_lane_error(cameras.csiFront.imageData)
                    if lane_error is not None:
                        lane_delta = float(
                            np.clip(
                                args.lane_gain * lane_error,
                                -args.lane_max_correction,
                                args.lane_max_correction,
                            )
                        )

                delta = float(np.clip(delta_map + lane_delta, -args.max_steer, args.max_steer))

                speed_gain_rules = 1.0
                stop_state = "disabled"
                stop_sign_dist = float("inf")
                person_dist = float("inf")
                person_blocking = False

                if detection_client is not None:
                    detection_client.read()
                    status = rule_machine.update(
                        detection_client.buffer[ROW_STOP_SIGN],
                        detection_client.buffer[ROW_PERSON],
                    )
                    speed_gain_rules = status.speed_gain
                    stop_state = status.stop_state
                    stop_sign_dist = status.stop_sign_distance_m
                    person_dist = status.person_distance_m
                    person_blocking = status.person_blocking

                corner_gain = np.clip(
                    1.0 - args.steer_slow_gain * abs(delta) / max(args.max_steer, 1e-3),
                    args.min_corner_gain,
                    1.0,
                )
                v_ref = args.cruise_speed * corner_gain * speed_gain_rules

                # If route is finished in non-cyclic mode, come to a controlled stop.
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
                        f"stop_state={stop_state} "
                        f"stop_d={stop_sign_dist:4.2f}m ped_d={person_dist:4.2f}m "
                        f"ped_block={int(person_blocking)}"
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
        stop_leds = np.zeros(8, dtype=np.float64)
        try:
            qcar.read_write_std(throttle=0.0, steering=0.0, LEDs=stop_leds)
        except Exception:
            pass

        if detection_client is not None:
            detection_client.terminate()
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
