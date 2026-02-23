#!/usr/bin/env python3
"""Single-file QCar2 autonomy: lane keeping + YOLO road rules + route following."""

import argparse
import heapq
import os
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
from hal.utilities.image_processing import ImageProcessing
from pal.products.qcar import IS_PHYSICAL_QCAR, QCar, QCarCameras, QCarGPS
from pal.utilities.math import Filter, wrap_to_pi
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


class SpeedController:
    # Matches the anti-surge controller behavior used in the referenced ACC_2026_Virtual repo.
    def __init__(self, kp=0.04, ki=0.15, max_throttle=0.14):
        self.maxThrottle = float(max_throttle)
        self.kp = float(kp)
        self.ki = float(ki)
        self.ei = 0.0

    def reset(self):
        self.ei = 0.0

    def update(self, v, v_ref, dt):
        e = float(v_ref - v)
        self.ei += max(float(dt), 1e-3) * e
        self.ei = float(np.clip(self.ei, -0.2, 0.2))
        if v_ref <= 0.01:
            self.ei *= 0.9
        u = self.kp * e + self.ki * self.ei
        return float(np.clip(u, 0.0, self.maxThrottle))


class PathSteeringController:
    def __init__(self, waypoints, k=0.6, cyclic=False, max_steer=0.52, switch_distance=0.5, search_window=60):
        self.maxSteeringAngle = float(max_steer)
        self.wp = waypoints
        self.N = int(waypoints.shape[1])
        self.wpi = 0
        self.k = float(k)
        self.cyclic = bool(cyclic)
        self.switchDistance = float(switch_distance)
        self.searchWindow = max(10, int(search_window))
        self.pathComplete = False

    def reanchor_to_position(self, p):
        if self.N < 2:
            self.wpi = 0
            self.pathComplete = True
            return
        p2 = np.asarray(p[:2], dtype=np.float64)
        pts = self.wp[:2, :].T
        self.wpi = int(np.argmin(np.linalg.norm(pts - p2, axis=1)))
        if not self.cyclic and self.wpi >= self.N - 1:
            self.wpi = self.N - 2
            self.pathComplete = True

    def _advance_to_local_nearest(self, p):
        if self.N < 2:
            return
        p2 = np.asarray(p[:2], dtype=np.float64)
        start = max(0, self.wpi - 2)
        end = min(self.N, self.wpi + self.searchWindow)
        if end - start < 2:
            return
        local_pts = self.wp[:2, start:end].T
        nearest_local = int(np.argmin(np.linalg.norm(local_pts - p2, axis=1)))
        nearest_idx = start + nearest_local
        if nearest_idx > self.wpi:
            self.wpi = nearest_idx

    def _advance_index(self):
        if self.cyclic:
            self.wpi = int((self.wpi + 1) % max(self.N - 1, 1))
            return
        if self.wpi < self.N - 2:
            self.wpi += 1
        else:
            self.pathComplete = True

    def update(self, p, th, speed):
        if self.N < 2:
            self.pathComplete = True
            return 0.0

        if not self.cyclic and self.wpi >= self.N - 2:
            self.pathComplete = True

        self._advance_to_local_nearest(p)

        i1 = int(self.wpi)
        if self.cyclic:
            i2 = int((i1 + 1) % max(self.N - 1, 1))
        else:
            i2 = min(i1 + 1, self.N - 1)

        wp_1 = self.wp[:2, i1]
        wp_2 = self.wp[:2, i2]
        p2 = np.asarray(p[:2], dtype=np.float64)

        v_seg = wp_2 - wp_1
        v_mag = float(np.linalg.norm(v_seg))
        if v_mag < 1e-6:
            self._advance_index()
            return 0.0

        v_uv = v_seg / v_mag
        tangent = float(np.arctan2(v_uv[1], v_uv[0]))
        s = float(np.dot(p2 - wp_1, v_uv))
        dist_to_next = float(np.linalg.norm(p2 - wp_2))

        switch_threshold = min(v_mag, max(self.switchDistance, 0.08))
        if s >= v_mag or dist_to_next < switch_threshold:
            self._advance_index()
        else:
            heading = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
            to_next = wp_2 - p2
            if float(np.dot(heading, to_next)) < -0.1:
                self._advance_index()

        s_clamped = float(np.clip(s, 0.0, v_mag))
        ep = wp_1 + v_uv * s_clamped
        ct = ep - p2
        side_dir = float(wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent))
        ect = float(np.linalg.norm(ct) * np.sign(side_dir))
        psi = float(wrap_to_pi(tangent - th))
        calc_speed = max(float(abs(speed)), 0.2)

        steering = psi + np.arctan2(self.k * ect, calc_speed)
        return float(np.clip(wrap_to_pi(steering), -self.maxSteeringAngle, self.maxSteeringAngle))


def parse_args():
    parser = argparse.ArgumentParser(description="QCar2 single-file autonomous driver")

    parser.add_argument("--model-path", default="", help="Path to ultralytics YOLO road-sign model")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--clipping-distance", type=float, default=10.0)

    parser.add_argument("--nodes", default="")
    parser.add_argument("--left-hand-traffic", action="store_true")
    parser.add_argument("--small-map", action="store_true")
    parser.add_argument("--route-cyclic", action="store_true")
    parser.add_argument("--auto-route-max-nodes", type=int, default=0)
    parser.add_argument("--auto-route-start-node", type=int, default=-1)
    parser.add_argument("--disable-opposite-turn-avoidance", action="store_true")
    parser.add_argument("--opposite-turn-threshold-deg", type=float, default=18.0)

    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument("--runtime", type=float, default=900.0)
    parser.add_argument("--start-delay", type=float, default=1.0)
    parser.add_argument("--cruise-speed", type=float, default=0.35)
    parser.add_argument("--min-speed", type=float, default=0.16)
    parser.add_argument("--max-throttle", type=float, default=0.14)
    parser.add_argument("--throttle-up-rate", type=float, default=0.65)
    parser.add_argument("--throttle-down-rate", type=float, default=2.2)
    parser.add_argument("--max-steer", type=float, default=0.52)
    parser.add_argument("--steering-k", "--stanley-gain", dest="steering_k", type=float, default=0.6)
    parser.add_argument("--speed-kp", type=float, default=0.04)
    parser.add_argument("--speed-ki", type=float, default=0.15)
    parser.add_argument("--steer-slow-gain", type=float, default=0.55)
    parser.add_argument("--min-corner-gain", type=float, default=0.30)
    parser.add_argument("--curve-lookahead-points", type=int, default=24)
    parser.add_argument("--curve-point-step", type=int, default=3)
    parser.add_argument("--curve-lat-accel", type=float, default=0.32)
    parser.add_argument("--speed-up-rate", type=float, default=0.30)
    parser.add_argument("--speed-down-rate", type=float, default=1.80)
    parser.add_argument("--goal-slowdown-distance", type=float, default=1.8)
    parser.add_argument("--goal-stop-distance", type=float, default=0.35)
    parser.add_argument("--goal-min-speed", type=float, default=0.08)
    parser.add_argument("--node-arrival-radius", type=float, default=0.30)
    parser.add_argument("--waypoint-switch-distance", type=float, default=0.5)
    parser.add_argument("--waypoint-search-window", type=int, default=60)
    parser.add_argument("--steer-rate-limit", type=float, default=0.95)

    parser.add_argument("--disable-lane-centering", action="store_true")
    parser.add_argument("--lane-cutoff-hz", type=float, default=25.0)
    parser.add_argument("--lane-steer-limit", type=float, default=0.5)
    parser.add_argument("--lane-steer-weight", type=float, default=0.45)
    parser.add_argument("--map-steer-weight", type=float, default=0.90)
    parser.add_argument("--yellow-line-target-ratio", type=float, default=-1.0)
    parser.add_argument("--yellow-line-distance-gain", type=float, default=0.45)
    parser.add_argument("--yellow-line-bottom-roi", type=float, default=0.55)
    parser.add_argument("--lane-turn-disable-ratio", type=float, default=0.95)
    parser.add_argument("--lane-recover-threshold", type=float, default=0.18)
    parser.add_argument("--lane-recover-boost", type=float, default=1.6)
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
    parser.add_argument("--yield-gain", type=float, default=0.85)
    parser.add_argument("--yield-clear-car-distance", type=float, default=0.95)
    parser.add_argument("--yield-clear-obstacle-distance", type=float, default=0.60)
    parser.add_argument("--car-stop", type=float, default=0.40)
    parser.add_argument("--car-slow", type=float, default=1.20)
    parser.add_argument("--car-forward-x-min", type=float, default=0.25)
    parser.add_argument("--car-forward-x-max", type=float, default=0.75)
    parser.add_argument("--car-forward-y-min", type=float, default=0.28)
    parser.add_argument("--car-min-area-ratio", type=float, default=0.0012)
    parser.add_argument("--obstacle-stop", type=float, default=0.28)
    parser.add_argument("--obstacle-slow", type=float, default=0.85)
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
    parser.add_argument("--native-cleanup", action="store_true")
    parser.add_argument("--disable-skyview-printout", action="store_true")
    parser.add_argument("--skyview-output-dir", default="")
    parser.add_argument("--skyview-dpi", type=int, default=220)

    return parser.parse_args()


def parse_nodes(nodes_str):
    return [int(token.strip()) for token in nodes_str.split(",") if token.strip()]


def parse_pose(pose_str):
    values = [float(token.strip()) for token in pose_str.split(",") if token.strip()]
    if len(values) != 3:
        raise ValueError("calibration pose must contain exactly 3 comma-separated values")
    return values


def resolve_start_node(roadmap, start_pose, start_node_override):
    n_nodes = len(roadmap.nodes)
    if n_nodes <= 0:
        raise RuntimeError("Roadmap has no nodes.")

    if int(start_node_override) >= 0 and int(start_node_override) < n_nodes:
        return int(start_node_override)

    if not hasattr(roadmap, "get_closest_node"):
        raise RuntimeError("HAL RoadMap.get_closest_node is required for dynamic routing.")

    idx = int(roadmap.get_closest_node(np.asarray(start_pose, dtype=np.float64)))
    if idx < 0 or idx >= n_nodes:
        raise RuntimeError("HAL get_closest_node returned an invalid node index.")
    return idx


def edge_initial_heading(edge):
    p0 = np.asarray(edge.fromNode.pose[:2, 0], dtype=np.float64)

    if edge.waypoints is not None and edge.waypoints.shape[1] > 0:
        horizon = min(edge.waypoints.shape[1], 24)
        for j in range(horizon):
            pj = np.asarray(edge.waypoints[:2, j], dtype=np.float64)
            d = pj - p0
            if np.linalg.norm(d) > 1e-4:
                return float(np.arctan2(d[1], d[0]))

    p1 = np.asarray(edge.toNode.pose[:2, 0], dtype=np.float64)
    d = p1 - p0
    if np.linalg.norm(d) > 1e-4:
        return float(np.arctan2(d[1], d[0]))
    return None


def edge_requires_opposite_turn(roadmap, start_node_idx, edge, left_hand_traffic, threshold_rad):
    start_pose = roadmap.get_node_pose(int(start_node_idx)).squeeze()
    start_heading = float(start_pose[2])
    departure_heading = edge_initial_heading(edge)
    if departure_heading is None:
        return False
    delta = float(wrap_to_pi(departure_heading - start_heading))
    if left_hand_traffic:
        return delta < -abs(float(threshold_rad))
    return delta > abs(float(threshold_rad))


def segment_length(path_segment):
    if path_segment is None or path_segment.shape[1] < 2:
        return float("inf")
    delta = np.diff(path_segment[:2, :], axis=1)
    return float(np.sum(np.linalg.norm(delta, axis=0)))


def find_shortest_path_with_turn_constraints(
    roadmap,
    start_node_idx,
    goal_node_idx,
    left_hand_traffic=False,
    avoid_opposite_turns=True,
    opposite_turn_threshold_rad=0.314,
):
    n_nodes = len(roadmap.nodes)
    start_node_idx = int(start_node_idx)
    goal_node_idx = int(goal_node_idx)
    if start_node_idx < 0 or goal_node_idx < 0 or start_node_idx >= n_nodes or goal_node_idx >= n_nodes:
        return None, None
    if start_node_idx == goal_node_idx:
        return None, [start_node_idx]

    nodes = roadmap.nodes
    node_to_index = {id(node): idx for idx, node in enumerate(nodes)}
    start_node = nodes[start_node_idx]
    goal_node = nodes[goal_node_idx]

    def _heuristic(node):
        return float(np.linalg.norm(goal_node.pose[:2, :] - node.pose[:2, :]))

    g_score = {node: float("inf") for node in nodes}
    g_score[start_node] = 0.0
    came_from = {node: None for node in nodes}

    open_set = []
    push_id = 0
    heapq.heappush(open_set, (_heuristic(start_node), push_id, start_node))
    closed = set()

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        if current is goal_node:
            break
        closed.add(current)

        current_idx = node_to_index[id(current)]
        for edge in current.outEdges:
            if edge.length is None:
                continue
            if (
                avoid_opposite_turns
                and edge_requires_opposite_turn(
                    roadmap=roadmap,
                    start_node_idx=current_idx,
                    edge=edge,
                    left_hand_traffic=left_hand_traffic,
                    threshold_rad=opposite_turn_threshold_rad,
                )
            ):
                continue

            neighbor = edge.toNode
            tentative_g = g_score[current] + float(edge.length)
            if tentative_g + 1e-9 >= g_score[neighbor]:
                continue
            came_from[neighbor] = (current, edge)
            g_score[neighbor] = tentative_g
            push_id += 1
            f_score = tentative_g + _heuristic(neighbor)
            heapq.heappush(open_set, (f_score, push_id, neighbor))

    if came_from[goal_node] is None:
        return None, None

    path = goal_node.pose[:2, :]
    node_path = [goal_node_idx]
    node = goal_node
    while came_from[node] is not None:
        prev_node, edge = came_from[node]
        if edge.waypoints is not None and edge.waypoints.shape[1] > 0:
            path = np.hstack((prev_node.pose[:2, :], edge.waypoints, path))
        else:
            path = np.hstack((prev_node.pose[:2, :], path))
        node = prev_node
        node_path.append(node_to_index[id(node)])
    node_path.reverse()
    return path, node_path


def generate_path_with_turn_constraints(
    roadmap,
    node_sequence,
    left_hand_traffic=False,
    avoid_opposite_turns=True,
    opposite_turn_threshold_rad=0.314,
):
    assert isinstance(node_sequence, (list, tuple)), "Node sequence must be a list/tuple."
    path = np.empty((2, 0))
    for i in range(len(node_sequence) - 1):
        start_node = int(node_sequence[i])
        goal_node = int(node_sequence[i + 1])

        segment, _ = find_shortest_path_with_turn_constraints(
            roadmap=roadmap,
            start_node_idx=start_node,
            goal_node_idx=goal_node,
            left_hand_traffic=left_hand_traffic,
            avoid_opposite_turns=avoid_opposite_turns,
            opposite_turn_threshold_rad=opposite_turn_threshold_rad,
        )
        used_fallback = False
        if segment is None and avoid_opposite_turns:
            segment, _ = find_shortest_path_with_turn_constraints(
                roadmap=roadmap,
                start_node_idx=start_node,
                goal_node_idx=goal_node,
                left_hand_traffic=left_hand_traffic,
                avoid_opposite_turns=False,
                opposite_turn_threshold_rad=opposite_turn_threshold_rad,
            )
            used_fallback = segment is not None

        if segment is None:
            return None
        if used_fallback:
            print(
                f"Warning: route leg {start_node}->{goal_node} has no opposite-turn-safe path. "
                "Using unconstrained fallback."
            )
        path = np.hstack((path, segment[:, :-1]))
    return path


def build_dynamic_node_sequence(
    roadmap,
    start_pose,
    route_cyclic,
    max_nodes,
    start_node_override,
    left_hand_traffic=False,
    avoid_opposite_turns=True,
    opposite_turn_threshold_rad=0.314,
):
    # Build route goals using HAL RoadMap APIs only:
    # - resolve start with get_closest_node
    # - test traversability through roadmap nodes/outEdges with A* search
    n_nodes = len(roadmap.nodes)
    if n_nodes <= 1:
        raise RuntimeError("Dynamic route generation failed: map needs at least 2 nodes.")

    start_node = resolve_start_node(roadmap, start_pose, start_node_override)

    sequence = [start_node]
    unvisited = set(range(n_nodes))
    unvisited.discard(start_node)
    current = start_node
    max_count = int(max_nodes) if int(max_nodes) > 1 else n_nodes

    while unvisited and len(sequence) < max_count:
        best_node = None
        best_cost = float("inf")
        best_any_node = None
        best_any_cost = float("inf")
        best_any_forbidden = False
        for candidate in list(unvisited):
            segment = None
            if avoid_opposite_turns:
                segment, _ = find_shortest_path_with_turn_constraints(
                    roadmap=roadmap,
                    start_node_idx=int(current),
                    goal_node_idx=int(candidate),
                    left_hand_traffic=left_hand_traffic,
                    avoid_opposite_turns=True,
                    opposite_turn_threshold_rad=opposite_turn_threshold_rad,
                )
                if segment is None:
                    try:
                        segment = roadmap.find_shortest_path(current, candidate)
                    except Exception:
                        segment = None
                    if segment is not None:
                        cost_any = segment_length(segment)
                        if cost_any < best_any_cost:
                            best_any_cost = cost_any
                            best_any_node = candidate
                            best_any_forbidden = True
                    continue
            else:
                try:
                    segment = roadmap.find_shortest_path(current, candidate)
                except Exception:
                    segment = None

            if segment is None:
                continue
            cost = segment_length(segment)
            if cost < best_any_cost:
                best_any_cost = cost
                best_any_node = candidate
                best_any_forbidden = False
            if cost < best_cost:
                best_cost = cost
                best_node = candidate
        if best_node is None:
            if best_any_node is None:
                break
            best_node = best_any_node
            if best_any_forbidden:
                print(
                    f"Warning: no opposite-turn-safe option from node {current}. "
                    f"Using fallback node {best_any_node}."
                )
        sequence.append(best_node)
        unvisited.discard(best_node)
        current = best_node

    if len(sequence) < 2:
        node_index = {id(node): idx for idx, node in enumerate(roadmap.nodes)}
        for edge in roadmap.nodes[start_node].outEdges:
            next_idx = node_index.get(id(edge.toNode), None)
            if next_idx is None or next_idx == start_node:
                continue
            sequence.append(next_idx)
            break

    if route_cyclic and len(sequence) >= 2:
        if avoid_opposite_turns:
            back_segment, _ = find_shortest_path_with_turn_constraints(
                roadmap=roadmap,
                start_node_idx=int(sequence[-1]),
                goal_node_idx=int(sequence[0]),
                left_hand_traffic=left_hand_traffic,
                avoid_opposite_turns=True,
                opposite_turn_threshold_rad=opposite_turn_threshold_rad,
            )
            back_forbidden = back_segment is None
        else:
            try:
                back_segment = roadmap.find_shortest_path(sequence[-1], sequence[0])
            except Exception:
                back_segment = None
            back_forbidden = False

        if back_segment is not None and not back_forbidden:
            sequence.append(sequence[0])
        elif back_forbidden:
            print(
                f"Warning: cyclic close from node {sequence[-1]} to {sequence[0]} "
                "requires opposite turn; leaving route non-cyclic."
            )

    if len(sequence) < 2:
        raise RuntimeError("Dynamic route generation failed: no traversable neighbor from start node.")

    return sequence


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


def compute_lane_steering_example(
    front_bgr,
    dt,
    steering_filter,
    lane_steer_limit,
    left_hand_traffic=False,
    yellow_line_target_ratio=-1.0,
    yellow_line_distance_gain=0.45,
    yellow_line_bottom_roi=0.55,
):
    h, w = front_bgr.shape[:2]
    row_start = int(np.clip(round((524.0 / 820.0) * h), 0, h - 1))
    row_end = int(np.clip(round((674.0 / 820.0) * h), row_start + 1, h))
    col_span = int(np.clip(round((820.0 / 1640.0) * w), 1, w))
    if left_hand_traffic:
        col_start = 0
        col_end = col_span
    else:
        col_start = max(0, w - col_span)
        col_end = w
    cropped = front_bgr[row_start:row_end, col_start:col_end]
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

    target_ratio = float(yellow_line_target_ratio)
    if target_ratio < 0.0:
        target_ratio = 0.70 if left_hand_traffic else 0.30

    rows, cols = np.where(binary > 0)
    if rows.size > 20:
        cutoff_row = int(np.clip(round(yellow_line_bottom_roi * binary.shape[0]), 0, binary.shape[0] - 1))
        near_mask = rows >= cutoff_row
        if np.count_nonzero(near_mask) >= 10:
            line_x = float(np.median(cols[near_mask]))
        else:
            line_x = float(np.median(cols))
        target_x = target_ratio * float(max(binary.shape[1] - 1, 1))
        x_error_norm = (line_x - target_x) / float(max(binary.shape[1], 1))
        raw_steering += -float(yellow_line_distance_gain) * x_error_norm

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
    # Narrow, forward-looking ROI to reduce false stops from sidewalks/curbs.
    y1 = int(0.35 * h)
    y2 = int(0.72 * h)
    x1 = int(0.42 * w)
    x2 = int(0.58 * w)
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
        "obstacle": float("inf"),
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
        img_h = max(1, int(rgb.shape[0]))
        img_w = max(1, int(rgb.shape[1]))
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

            if mapped == "car":
                cx = (x1 + x2) / (2.0 * img_w)
                cy = (y1 + y2) / (2.0 * img_h)
                area_ratio = ((x2 - x1) * (y2 - y1)) / float(img_w * img_h)
                in_forward_fov = args.car_forward_x_min <= cx <= args.car_forward_x_max and cy >= args.car_forward_y_min
                if not in_forward_fov or area_ratio < args.car_min_area_ratio:
                    continue

            d = median_depth_from_bbox(depth, (x1, y1, x2, y2), inset=0.2)
            if not np.isfinite(d) or d <= 0 or d > args.clipping_distance:
                continue
            distances[mapped] = min(distances[mapped], d)

    obs_d = forward_obstacle_distance(depth)
    if obs_d < args.forward_obstacle_max:
        distances["obstacle"] = min(distances["obstacle"], obs_d)

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

    # Yield means slow/merge, not full stop, unless there is actual conflicting traffic.
    if dists["yield"] < args.yield_trigger:
        conflicting = (
            dists["car"] < args.yield_clear_car_distance
            or state.person_blocking
        )
        if conflicting:
            gain *= float(np.clip(args.yield_gain, 0.0, 1.0))

    return float(np.clip(gain, 0.0, 1.0))


def rate_limit(value, target, dt, up_rate, down_rate):
    max_up = max(float(up_rate), 1e-3) * dt
    max_down = max(float(down_rate), 1e-3) * dt
    if target >= value:
        return min(target, value + max_up)
    return max(target, value - max_down)


def curvature_from_points(p0, p1, p2):
    a = float(np.linalg.norm(p1 - p0))
    b = float(np.linalg.norm(p2 - p1))
    c = float(np.linalg.norm(p2 - p0))
    if min(a, b, c) < 1e-4:
        return 0.0
    twice_area = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    return float((2.0 * twice_area) / (a * b * c))


def compute_curve_speed_limit(start_idx, waypoints, route_cyclic, args):
    if waypoints is None or waypoints.shape[1] < 3:
        return float(args.cruise_speed), 0.0

    pts = waypoints[:2, :].T
    n_pts = int(pts.shape[0])
    if n_pts < 3:
        return float(args.cruise_speed), 0.0
    base_idx = int(np.clip(start_idx, 0, n_pts - 1))

    step = max(1, int(args.curve_point_step))
    horizon = max(3, int(args.curve_lookahead_points))
    max_kappa = 0.0

    for offset in range(0, horizon, step):
        i0 = base_idx + offset
        i1 = i0 + step
        i2 = i1 + step

        if route_cyclic:
            i0 %= n_pts
            i1 %= n_pts
            i2 %= n_pts
        elif i2 >= n_pts:
            break

        kappa = curvature_from_points(pts[i0], pts[i1], pts[i2])
        if kappa > max_kappa:
            max_kappa = kappa

    if max_kappa <= 1e-6:
        speed_limit = float(args.cruise_speed)
    else:
        speed_limit = float(np.sqrt(max(float(args.curve_lat_accel), 0.01) / max_kappa))

    min_speed = float(np.clip(args.min_speed, 0.0, args.cruise_speed))
    speed_limit = float(np.clip(speed_limit, min_speed, args.cruise_speed))
    return speed_limit, max_kappa


def remaining_path_distance(waypoints, start_idx):
    if waypoints is None or waypoints.shape[1] < 2:
        return 0.0
    start = int(np.clip(start_idx, 0, waypoints.shape[1] - 1))
    tail = waypoints[:2, start:]
    if tail.shape[1] < 2:
        return 0.0
    seg = np.diff(tail, axis=1)
    return float(np.sum(np.linalg.norm(seg, axis=0)))


def emit_route_skyview(roadmap, node_sequence, waypoint_sequence, route_source, args):
    if args.disable_skyview_printout:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:
        pass

    try:
        output_dir = Path(args.skyview_output_dir).expanduser() if args.skyview_output_dir else (Path(__file__).resolve().parent / "skyview")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_output_path = output_dir / f"route_skyview_{timestamp}.png"
        latest_output_path = output_dir / "route_skyview_latest.png"

        plt_obj, ax = roadmap.display()

        if waypoint_sequence is not None and waypoint_sequence.size > 0:
            ax.plot(
                waypoint_sequence[0, :],
                waypoint_sequence[1, :],
                color="cyan",
                linewidth=2.4,
                linestyle="-",
                zorder=3,
                label="planned_path",
            )

        route_xy = []
        for idx, node_id in enumerate(node_sequence):
            pose = roadmap.get_node_pose(int(node_id)).squeeze()
            x = float(pose[0])
            y = float(pose[1])
            route_xy.append([x, y])
            ax.text(
                x + 0.03,
                y - 0.04,
                f"{idx}:{int(node_id)}",
                fontsize=8,
                color="yellow",
                zorder=4,
            )

        if route_xy:
            route_xy = np.asarray(route_xy, dtype=np.float64)
            ax.plot(
                route_xy[:, 0],
                route_xy[:, 1],
                color="magenta",
                linewidth=1.8,
                linestyle="--",
                marker="o",
                markersize=3.5,
                zorder=4,
                label="node_sequence",
            )

        ax.set_title(f"QCar2 Route Skyview ({route_source})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

        plt_obj.savefig(str(run_output_path), dpi=int(max(args.skyview_dpi, 80)), bbox_inches="tight")
        plt_obj.savefig(str(latest_output_path), dpi=int(max(args.skyview_dpi, 80)), bbox_inches="tight")
        plt_obj.close(ax.figure)

        print(f"Skyview map saved: {run_output_path}")
        print(f"Skyview latest: {latest_output_path}")
        print("Skyview node printout:")
        for idx, node_id in enumerate(node_sequence):
            pose = roadmap.get_node_pose(int(node_id)).squeeze()
            print(f"  [{idx:02d}] node {int(node_id):02d} -> x={float(pose[0]): .3f}, y={float(pose[1]): .3f}, th={float(pose[2]): .3f}")

    except Exception as exc:
        print(f"Warning: could not generate skyview route printout: {exc}")


def main():
    global STOP_REQUESTED
    args = parse_args()
    signal.signal(signal.SIGINT, signal_handler)

    if args.qcar_id < 0 or args.virtual_port_stride < 0:
        raise ValueError("qcar-id and virtual-port-stride must be >= 0")
    if args.cruise_speed <= 0:
        raise ValueError("cruise-speed must be > 0")
    if args.min_speed < 0:
        raise ValueError("min-speed must be >= 0")
    if args.max_throttle <= 0:
        raise ValueError("max-throttle must be > 0")
    if args.max_steer <= 0:
        raise ValueError("max-steer must be > 0")
    if args.steer_rate_limit <= 0:
        raise ValueError("steer-rate-limit must be > 0")
    if args.lane_turn_disable_ratio <= 0:
        raise ValueError("lane-turn-disable-ratio must be > 0")
    if args.yellow_line_target_ratio > 1.0:
        raise ValueError("yellow-line-target-ratio must be <= 1.0 (or negative for auto)")
    if args.yellow_line_distance_gain < 0:
        raise ValueError("yellow-line-distance-gain must be >= 0")
    if args.yellow_line_bottom_roi < 0 or args.yellow_line_bottom_roi > 1:
        raise ValueError("yellow-line-bottom-roi must be in [0, 1]")
    if not (0.0 <= args.car_forward_x_min < args.car_forward_x_max <= 1.0):
        raise ValueError("car-forward-x-min/max must satisfy 0 <= min < max <= 1")
    if not (0.0 <= args.car_forward_y_min <= 1.0):
        raise ValueError("car-forward-y-min must be in [0, 1]")
    if args.car_min_area_ratio < 0:
        raise ValueError("car-min-area-ratio must be >= 0")
    if args.auto_route_max_nodes < 0:
        raise ValueError("auto-route-max-nodes must be >= 0")
    if args.auto_route_start_node < -1:
        raise ValueError("auto-route-start-node must be >= -1")
    if args.goal_stop_distance <= 0 or args.goal_slowdown_distance <= 0:
        raise ValueError("goal-stop-distance and goal-slowdown-distance must be > 0")
    if args.node_arrival_radius <= 0:
        raise ValueError("node-arrival-radius must be > 0")
    if args.goal_min_speed < 0:
        raise ValueError("goal-min-speed must be >= 0")
    if args.opposite_turn_threshold_deg < 0 or args.opposite_turn_threshold_deg > 180:
        raise ValueError("opposite-turn-threshold-deg must be in [0, 180]")
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

    calibration_pose = parse_pose(args.calibration_pose)

    roadmap = SDCSRoadMap(leftHandTraffic=args.left_hand_traffic, useSmallMap=args.small_map)
    requested_nodes = parse_nodes(args.nodes)
    if len(requested_nodes) >= 2:
        node_sequence = requested_nodes
        route_source = "manual"
    else:
        node_sequence = build_dynamic_node_sequence(
            roadmap=roadmap,
            start_pose=np.array(calibration_pose, dtype=np.float64),
            route_cyclic=args.route_cyclic,
            max_nodes=args.auto_route_max_nodes,
            start_node_override=args.auto_route_start_node,
            left_hand_traffic=args.left_hand_traffic,
            avoid_opposite_turns=(not args.disable_opposite_turn_avoidance),
            opposite_turn_threshold_rad=np.deg2rad(args.opposite_turn_threshold_deg),
        )
        route_source = "dynamic"

    if args.disable_opposite_turn_avoidance:
        try:
            waypoint_sequence = roadmap.generate_path(node_sequence)
        except TypeError:
            waypoint_sequence = roadmap.generate_path(nodeSequence=node_sequence)
    else:
        waypoint_sequence = generate_path_with_turn_constraints(
            roadmap=roadmap,
            node_sequence=node_sequence,
            left_hand_traffic=args.left_hand_traffic,
            avoid_opposite_turns=True,
            opposite_turn_threshold_rad=np.deg2rad(args.opposite_turn_threshold_deg),
        )
    if waypoint_sequence is None:
        raise RuntimeError("Could not generate a route for the provided node sequence.")

    route_node_xy = np.array(
        [roadmap.get_node_pose(int(node_id)).squeeze()[:2] for node_id in node_sequence],
        dtype=np.float64,
    )

    initial_pose = roadmap.get_node_pose(node_sequence[0]).squeeze()
    steering_controller = PathSteeringController(
        waypoints=waypoint_sequence,
        k=args.steering_k,
        cyclic=args.route_cyclic,
        max_steer=args.max_steer,
        switch_distance=args.waypoint_switch_distance,
        search_window=args.waypoint_search_window,
    )
    speed_controller = SpeedController(kp=args.speed_kp, ki=args.speed_ki, max_throttle=args.max_throttle)
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
    print("Route source:", route_source)
    print("Route nodes:", node_sequence)
    print("Route node count:", len(node_sequence))
    print(
        "Opposite-turn avoidance:",
        (not args.disable_opposite_turn_avoidance),
        f"(threshold={args.opposite_turn_threshold_deg:.1f} deg)",
    )
    print("Running on physical QCar:", IS_PHYSICAL_QCAR)
    print("QLabs actor ID:", args.qcar_id)
    print("Model:", args.model_path)
    print(
        "Yellow-line hold:",
        f"target_ratio={args.yellow_line_target_ratio:.2f}",
        f"gain={args.yellow_line_distance_gain:.2f}",
        f"bottom_roi={args.yellow_line_bottom_roi:.2f}",
    )
    print("Cruise/min speed:", f"{args.cruise_speed:.2f}/{min(args.min_speed, args.cruise_speed):.2f} m/s")
    emit_route_skyview(roadmap, node_sequence, waypoint_sequence, route_source, args)

    start_node_reached = True
    init_steering_controller = None
    init_waypoints = None
    node_progress_index = 0
    next_node_index = 1 if len(node_sequence) > 1 else 0

    try:
        t_wait = time.time()
        while not STOP_REQUESTED and time.time() - t_wait < 8.0:
            if gps.readGPS():
                init_pose = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                start_node_reached, init_waypoints = roadmap.initial_check(init_pose, node_sequence, waypoint_sequence)
                if not start_node_reached and init_waypoints is not None:
                    init_steering_controller = PathSteeringController(
                        waypoints=init_waypoints,
                        k=args.steering_k,
                        cyclic=False,
                        max_steer=args.max_steer,
                        switch_distance=args.waypoint_switch_distance,
                        search_window=args.waypoint_search_window,
                    )
                break
            time.sleep(0.02)

        dt_target = 1.0 / args.sample_rate
        t0 = time.time()
        t_prev = t0
        next_print = t0
        delta = 0.0
        v_plan = 0.0

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
            while next_node_index < len(node_sequence):
                dist_to_next_node = float(np.linalg.norm(route_node_xy[next_node_index] - p[:2]))
                if dist_to_next_node > args.node_arrival_radius:
                    break
                node_progress_index = next_node_index
                next_node_index += 1

            goal_dist = float(np.linalg.norm(route_node_xy[-1] - p[:2]))
            active_controller = steering_controller
            active_waypoints = waypoint_sequence

            if loop_start - t0 < args.start_delay:
                throttle = 0.0
                delta = 0.0
                v_plan = 0.0
                speed_controller.reset()
                dists = {
                    "stop": np.inf,
                    "person": np.inf,
                    "car": np.inf,
                    "obstacle": np.inf,
                    "yield": np.inf,
                    "traffic_red": np.inf,
                }
                rule_gain = 1.0
                curve_speed_limit = float(args.cruise_speed)
                max_kappa = 0.0
                remaining_dist = remaining_path_distance(waypoint_sequence, steering_controller.wpi)
                route_done = False
            else:
                v_for_steering = max(abs(v), 0.05)
                if not start_node_reached and init_steering_controller is not None:
                    dist_to_start = np.linalg.norm(waypoint_sequence[:, 0] - p)
                    start_node_reached = dist_to_start < 0.2
                    if start_node_reached:
                        steering_controller.reanchor_to_position(p)
                        delta_map = steering_controller.update(p, th, v_for_steering)
                    else:
                        active_controller = init_steering_controller
                        if init_waypoints is not None:
                            active_waypoints = init_waypoints
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
                        left_hand_traffic=args.left_hand_traffic,
                        yellow_line_target_ratio=args.yellow_line_target_ratio,
                        yellow_line_distance_gain=args.yellow_line_distance_gain,
                        yellow_line_bottom_roi=args.yellow_line_bottom_roi,
                    )
                lane_disabled_for_turn = abs(delta_map) > (args.lane_turn_disable_ratio * args.max_steer)
                if lane_steer is None or lane_disabled_for_turn:
                    delta_target = float(np.clip(delta_map, -args.max_steer, args.max_steer))
                else:
                    map_w = float(args.map_steer_weight)
                    lane_w = float(args.lane_steer_weight)
                    if abs(lane_steer - delta_map) > args.lane_recover_threshold:
                        lane_w *= float(args.lane_recover_boost)
                    norm = max(map_w + lane_w, 1e-3)
                    delta_target = float(np.clip((map_w * delta_map + lane_w * lane_steer) / norm, -args.max_steer, args.max_steer))
                delta = float(np.clip(rate_limit(delta, delta_target, dt, args.steer_rate_limit, args.steer_rate_limit), -args.max_steer, args.max_steer))

                dists = detect_distances(model, depth_rgb.rgb, depth_rgb.depth, args)
                rule_gain = compute_rule_gain(loop_start, dists, rule_state, args)

                steer_gain = np.clip(
                    1.0 - args.steer_slow_gain * abs(delta) / max(args.max_steer, 1e-3),
                    args.min_corner_gain,
                    1.0,
                )
                curve_speed_limit, max_kappa = compute_curve_speed_limit(
                    active_controller.wpi,
                    active_waypoints,
                    active_controller.cyclic,
                    args,
                )
                speed_target = min(float(args.cruise_speed * steer_gain), curve_speed_limit)

                remaining_dist = remaining_path_distance(waypoint_sequence, steering_controller.wpi)
                if not args.route_cyclic and remaining_dist < args.goal_slowdown_distance:
                    ratio = np.clip(remaining_dist / max(args.goal_slowdown_distance, 1e-3), 0.0, 1.0)
                    goal_speed_limit = args.goal_min_speed + (args.cruise_speed - args.goal_min_speed) * ratio
                    speed_target = min(speed_target, float(goal_speed_limit))

                final_node_reached = node_progress_index >= (len(node_sequence) - 1)
                route_done = (
                    (not args.route_cyclic)
                    and final_node_reached
                    and (goal_dist < max(args.goal_stop_distance, args.node_arrival_radius))
                )
                if route_done:
                    speed_target = 0.0

                v_plan = rate_limit(v_plan, speed_target, dt, args.speed_up_rate, args.speed_down_rate)
                v_ref = v_plan * rule_gain

                throttle_target = float(speed_controller.update(v, v_ref, dt))
                if v_ref <= 0.01 and v > 0.05:
                    throttle_target = -0.12
                elif v_ref <= 0.01:
                    throttle_target = 0.0

                throttle = rate_limit(throttle, throttle_target, dt, args.throttle_up_rate, args.throttle_down_rate)
                throttle = float(np.clip(throttle, -args.max_throttle, args.max_throttle))

                if args.print_rate > 0 and loop_start >= next_print:
                    next_print = loop_start + (1.0 / args.print_rate)
                    if next_node_index < len(node_sequence):
                        next_target_node = int(node_sequence[next_node_index])
                        next_target_dist = float(np.linalg.norm(route_node_xy[next_node_index] - p[:2]))
                    else:
                        next_target_node = int(node_sequence[-1])
                        next_target_dist = goal_dist
                    print(
                        f"t={loop_start - t0:6.1f}s "
                        f"v={v:4.2f}m/s v_ref={v_ref:4.2f} "
                        f"thr={throttle:5.2f} str={delta:5.2f} "
                        f"wp={steering_controller.wpi:4d}/{steering_controller.N - 1:4d} rem={remaining_dist:.2f} "
                        f"nodes={node_progress_index + 1:2d}/{len(node_sequence):2d} "
                        f"next={next_target_node:02d}@{next_target_dist:.2f} "
                        f"curve={curve_speed_limit:.2f} kappa={max_kappa:.3f} "
                        f"stop={dists['stop']:.2f} ped={dists['person']:.2f} "
                        f"car={dists['car']:.2f} obs={dists['obstacle']:.2f} red={dists['traffic_red']:.2f} "
                        f"yield={dists['yield']:.2f} gain={rule_gain:.2f}"
                    )

            leds = compute_leds(throttle, delta)
            qcar.read_write_std(throttle=throttle, steering=delta, LEDs=leds)

            if route_done and not args.route_cyclic and abs(v) < 0.03:
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
        if args.native_cleanup:
            try:
                depth_rgb.terminate()
            except Exception:
                pass
            try:
                cameras.terminate()
            except Exception:
                pass
            try:
                gps.terminate()
            except Exception:
                pass
            try:
                qcar.terminate()
            except Exception:
                pass
            if qlabs_client is not None:
                try:
                    qlabs_client.close()
                except Exception:
                    pass


if __name__ == "__main__":
    _ok = False
    try:
        main()
        _ok = True
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # Avoid post-run native destructor segfaults seen in containerized PAL/HAL stacks.
        if _ok:
            os._exit(0)
