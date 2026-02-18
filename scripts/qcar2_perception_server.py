#!/usr/bin/env python3
"""Perception stream server for QCar2 road-rule compliance.

Default mode uses custom OpenCV inference for pedestrians and stop signs,
then publishes the same detection packet format consumed by the driver.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from compat_qcar2_virtual import patch_qcar2_virtual_config

patch_qcar2_virtual_config()

from competition_common import (
    ROW_CAR,
    ROW_PERSON,
    ROW_STOP_SIGN,
    ROW_TRAFFIC_LIGHT,
    ROW_YIELD,
    DetectionStreamServer,
    append_detection_distance,
    empty_detection_packet,
)
from pit.YOLO.utils import QCar2DepthAligned

BASE_VIDEO3D_PORT = 18965


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QCar2 perception stream server")
    parser.add_argument("--stream-ip", default="localhost", help="IP for detection stream server")
    parser.add_argument("--stream-port", type=int, default=18666, help="TCP port for detection stream server")
    parser.add_argument("--depth-port", type=str, default="18777", help="Depth-align stream port")
    parser.add_argument(
        "--qcar-id",
        type=int,
        default=0,
        help="QLabs QCar2 actor number. Used to derive virtual sensor port defaults.",
    )
    parser.add_argument(
        "--virtual-port-stride",
        type=int,
        default=10,
        help="Port offset per QCar ID in virtual mode. Effective port = base + qcar_id*stride.",
    )
    parser.add_argument("--video3d-port", type=int, default=None, help="Override RealSense virtual port")
    parser.add_argument(
        "--inference-mode",
        choices=["custom", "yolo", "hybrid"],
        default="custom",
        help="Detection backend. 'custom' avoids YOLO dependency.",
    )
    parser.add_argument("--confidence", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument(
        "--person-score-threshold",
        type=float,
        default=0.35,
        help="HOG people detector score threshold (custom mode)",
    )
    parser.add_argument(
        "--person-nms-threshold",
        type=float,
        default=0.35,
        help="HOG people detector NMS IoU threshold (custom mode)",
    )
    parser.add_argument("--stop-min-area", type=float, default=450.0, help="Min red contour area for stop sign")
    parser.add_argument(
        "--stop-min-solidity",
        type=float,
        default=0.72,
        help="Min contour solidity for stop sign candidates",
    )
    parser.add_argument("--clipping-distance", type=float, default=10.0, help="Depth clipping distance (m)")
    parser.add_argument("--probe-ip", default="", help="If set, show annotated stream in an OpenCV window")
    parser.add_argument("--model-path", default="", help="Optional YOLO model path")
    return parser.parse_args()


def resolve_virtual_port(explicit_port: int | None, base_port: int, qcar_id: int, stride: int) -> int:
    if explicit_port is not None:
        return int(explicit_port)
    return int(base_port + qcar_id * stride)


def configure_virtual_depth_camera(depth_rgb: QCar2DepthAligned, video3d_port: int) -> None:
    if getattr(depth_rgb, "isPhysical", True):
        return
    if video3d_port == BASE_VIDEO3D_PORT:
        return

    try:
        from pal.utilities.vision import Camera3D
    except Exception as exc:
        print(f"Warning: could not import Camera3D for virtual port override: {exc}")
        return

    try:
        if hasattr(depth_rgb, "camera") and depth_rgb.camera is not None:
            depth_rgb.camera.terminate()
    except Exception:
        pass

    try:
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


def depth_plane(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3 and depth.shape[2] == 1:
        return depth[:, :, 0]
    return depth


def median_depth_from_bbox(depth: np.ndarray, bbox: tuple[int, int, int, int], inset: float = 0.2) -> float:
    d = depth_plane(depth)
    x, y, w, h = bbox
    x0 = max(0, int(x + w * inset))
    y0 = max(0, int(y + h * inset))
    x1 = min(d.shape[1], int(x + w * (1.0 - inset)))
    y1 = min(d.shape[0], int(y + h * (1.0 - inset)))
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    roi = d[y0:y1, x0:x1]
    valid = roi[np.isfinite(roi)]
    valid = valid[valid > 0]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def nms_boxes(
    boxes: list[tuple[int, int, int, int]], scores: list[float], iou_thresh: float
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    order = np.argsort(np.array(scores, dtype=np.float32))[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([bbox_iou(boxes[i], boxes[int(j)]) for j in rest], dtype=np.float32)
        order = rest[ious <= iou_thresh]
    return [boxes[i] for i in keep]


def red_mask(frame: np.ndarray) -> np.ndarray:
    hsv_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    def _mask(hsv: np.ndarray) -> np.ndarray:
        m1 = cv2.inRange(hsv, (0, 80, 50), (12, 255, 255))
        m2 = cv2.inRange(hsv, (160, 80, 50), (179, 255, 255))
        return cv2.bitwise_or(m1, m2)

    mask_a = _mask(hsv_bgr)
    mask_b = _mask(hsv_rgb)
    mask = mask_a if int(np.count_nonzero(mask_a)) >= int(np.count_nonzero(mask_b)) else mask_b

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_stop_signs_custom(
    frame: np.ndarray,
    depth: np.ndarray,
    min_area: float,
    min_solidity: float,
) -> tuple[list[float], list[tuple[int, int, int, int]]]:
    mask = red_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)
        if vertices < 6 or vertices > 10:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w <= 0 or h <= 0:
            continue
        ratio = w / float(h)
        if ratio < 0.65 or ratio > 1.35:
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 1e-3:
            continue
        solidity = area / hull_area
        if solidity < min_solidity:
            continue

        boxes.append((int(x), int(y), int(w), int(h)))
        scores.append(area)

    boxes = nms_boxes(boxes, scores, iou_thresh=0.4)
    distances: list[float] = []
    for bbox in boxes:
        dist = median_depth_from_bbox(depth, bbox, inset=0.3)
        if np.isfinite(dist) and dist > 0:
            distances.append(float(dist))
    return distances, boxes


def create_people_detector():
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog
    except Exception as exc:
        print(f"Warning: HOG people detector unavailable: {exc}")
        return None


def detect_people_custom(
    frame: np.ndarray,
    depth: np.ndarray,
    hog,
    score_threshold: float,
    nms_threshold: float,
) -> tuple[list[float], list[tuple[int, int, int, int]]]:
    if hog is None:
        return [], []

    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.03,
    )
    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    for (x, y, w, h), weight in zip(rects, weights):
        score = float(weight)
        if score < score_threshold:
            continue
        if h < 70 or w < 20:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
        scores.append(score)

    boxes = nms_boxes(boxes, scores, iou_thresh=nms_threshold)
    distances: list[float] = []
    for bbox in boxes:
        dist = median_depth_from_bbox(depth, bbox, inset=0.25)
        if np.isfinite(dist) and dist > 0:
            distances.append(float(dist))
    return distances, boxes


def build_custom_packet(stop_dists: list[float], person_dists: list[float]) -> np.ndarray:
    packet = empty_detection_packet()
    for d in sorted(stop_dists):
        append_detection_distance(packet[ROW_STOP_SIGN], d)
    for d in sorted(person_dists):
        append_detection_distance(packet[ROW_PERSON], d)
    return packet


def build_yolo_packet(processed_results: list) -> np.ndarray:
    packet = empty_detection_packet()
    for obj in processed_results:
        name = str(getattr(obj, "name", "")).lower()
        distance = float(getattr(obj, "distance", np.nan))

        if "stop sign" in name:
            append_detection_distance(packet[ROW_STOP_SIGN], distance)
        elif "traffic light" in name and "red" in name:
            append_detection_distance(packet[ROW_TRAFFIC_LIGHT], distance)
        elif "car" in name:
            append_detection_distance(packet[ROW_CAR], distance)
        elif "yield" in name:
            append_detection_distance(packet[ROW_YIELD], distance)
        elif "person" in name:
            append_detection_distance(packet[ROW_PERSON], distance)
    return packet


def merge_packets(dst: np.ndarray, src: np.ndarray) -> None:
    for row in range(src.shape[0]):
        count = int(src[row, 0])
        for col in range(1, count + 1):
            append_detection_distance(dst[row], float(src[row, col]))


def draw_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    label: str,
    color: tuple[int, int, int],
) -> None:
    for x, y, w, h in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image,
            label,
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    if args.qcar_id < 0:
        raise ValueError("--qcar-id must be >= 0")
    if args.virtual_port_stride < 0:
        raise ValueError("--virtual-port-stride must be >= 0")

    video3d_port = resolve_virtual_port(
        args.video3d_port,
        BASE_VIDEO3D_PORT,
        args.qcar_id,
        args.virtual_port_stride,
    )

    hog_detector = create_people_detector() if args.inference_mode in {"custom", "hybrid"} else None

    yolo_detector = None
    if args.inference_mode in {"yolo", "hybrid"}:
        try:
            from pit.YOLO.nets import YOLOv8

            yolo_detector = YOLOv8(
                imageHeight=480,
                imageWidth=640,
                modelPath=args.model_path or None,
            )
        except Exception as exc:
            if args.inference_mode == "yolo":
                raise RuntimeError(f"YOLO initialization failed: {exc}") from exc
            print(f"Warning: YOLO unavailable, continuing in custom-only mode: {exc}")

    depth_rgb = QCar2DepthAligned(port=args.depth_port)
    configure_virtual_depth_camera(depth_rgb, video3d_port)
    stream = DetectionStreamServer(ip=args.stream_ip, port=args.stream_port)

    show_window = bool(args.probe_ip)
    if show_window:
        cv2.namedWindow("qcar2_perception", cv2.WINDOW_NORMAL)

    print("Perception actor ID:", args.qcar_id)
    print("Perception video3d port:", video3d_port)
    print("Inference mode:", args.inference_mode if yolo_detector is not None else "custom")

    try:
        while True:
            if not depth_rgb.read():
                time.sleep(0.002)
                continue

            frame = depth_rgb.rgb
            depth = depth_rgb.depth
            packet = empty_detection_packet()
            annotated = frame.copy() if show_window else None

            if args.inference_mode in {"custom", "hybrid"} or yolo_detector is None:
                stop_dists, stop_boxes = detect_stop_signs_custom(
                    frame=frame,
                    depth=depth,
                    min_area=args.stop_min_area,
                    min_solidity=args.stop_min_solidity,
                )
                person_dists, person_boxes = detect_people_custom(
                    frame=frame,
                    depth=depth,
                    hog=hog_detector,
                    score_threshold=args.person_score_threshold,
                    nms_threshold=args.person_nms_threshold,
                )
                custom_packet = build_custom_packet(stop_dists, person_dists)
                merge_packets(packet, custom_packet)
                if annotated is not None:
                    draw_boxes(annotated, stop_boxes, "stop-sign", (0, 0, 255))
                    draw_boxes(annotated, person_boxes, "person", (0, 255, 0))

            if yolo_detector is not None:
                prepared = yolo_detector.pre_process(frame)
                yolo_detector.predict(
                    inputImg=prepared,
                    classes=[0, 2, 9, 11, 33],  # person, car, traffic light, stop sign, yield sign
                    confidence=args.confidence,
                    verbose=False,
                    half=True,
                )
                processed = yolo_detector.post_processing(
                    alignedDepth=depth,
                    clippingDistance=args.clipping_distance,
                )
                yolo_packet = build_yolo_packet(processed)
                if args.inference_mode == "yolo":
                    packet = yolo_packet
                else:
                    merge_packets(packet, yolo_packet)
                if annotated is not None and args.inference_mode == "yolo":
                    annotated = yolo_detector.post_process_render(showFPS=True)

            stream.send(packet)

            if annotated is not None:
                cv2.putText(
                    annotated,
                    f"mode: {args.inference_mode}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("qcar2_perception", annotated)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

    except KeyboardInterrupt:
        pass
    finally:
        if show_window:
            cv2.destroyAllWindows()
        depth_rgb.terminate()
        stream.terminate()


if __name__ == "__main__":
    main()
