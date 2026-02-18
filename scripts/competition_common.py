#!/usr/bin/env python3
"""Common helpers for QCar2 competition autonomy scripts."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

try:
    from quanser.common import Timeout
except Exception:  # pragma: no cover
    from quanser.communications import Timeout

from pal.utilities.stream import BasicStream


DETECTION_ROWS = 5
DETECTION_COLS = 7
ROW_STOP_SIGN = 0
ROW_TRAFFIC_LIGHT = 1
ROW_CAR = 2
ROW_YIELD = 3
ROW_PERSON = 4


def empty_detection_packet() -> np.ndarray:
    """Create a packet shaped like the YOLO detection stream payload."""
    return np.zeros((DETECTION_ROWS, DETECTION_COLS), dtype=np.float64)


def append_detection_distance(row: np.ndarray, distance_m: float) -> None:
    """Append a distance measurement into a detection row in-place."""
    if not np.isfinite(distance_m) or distance_m <= 0:
        return
    count = int(row[0])
    if count >= DETECTION_COLS - 1:
        return
    row[count + 1] = distance_m
    row[0] = count + 1


def closest_detection_distance(row: np.ndarray) -> float:
    """Return closest valid distance from a detection row, or +inf if none."""
    count = int(row[0])
    if count <= 0:
        return float("inf")
    distances = row[1 : count + 1]
    distances = distances[np.isfinite(distances)]
    distances = distances[distances > 0]
    if distances.size == 0:
        return float("inf")
    return float(np.min(distances))


class DetectionStreamClient:
    """Receive detection packets from the perception server."""

    def __init__(
        self,
        ip: str = "localhost",
        port: int = 18666,
        non_blocking: bool = True,
        connect_iterations: int = 40,
    ) -> None:
        self.uri = f"tcpip://{ip}:{port}"
        self._timeout = Timeout(seconds=0, nanoseconds=100_000)
        self._handle = BasicStream(
            uri=self.uri,
            agent="C",
            receiveBuffer=empty_detection_packet(),
            recvBufferSize=DETECTION_ROWS * DETECTION_COLS * 8,
            nonBlocking=non_blocking,
            reshapeOrder="C",
        )
        self.buffer = empty_detection_packet()
        self._connect(connect_iterations)

    def _connect(self, iterations: int) -> None:
        timeout = Timeout(seconds=0, nanoseconds=100_000)
        for _ in range(iterations):
            if self._handle.connected:
                return
            self._handle.checkConnection(timeout=timeout)
            if self._handle.connected:
                return

    @property
    def connected(self) -> bool:
        return bool(self._handle.connected)

    def read(self) -> bool:
        """Read one detection packet if available."""
        if not self._handle.connected:
            self._connect(1)
            return False

        recv_flag, _ = self._handle.receive(
            timeout=Timeout(seconds=0, nanoseconds=10_000),
            iterations=5,
        )
        if recv_flag:
            self.buffer[:] = self._handle.receiveBuffer
        return bool(recv_flag)

    def terminate(self) -> None:
        self._handle.terminate()

    def __enter__(self) -> "DetectionStreamClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.terminate()


class DetectionStreamServer:
    """Publish detection packets to the driving controller."""

    def __init__(
        self,
        ip: str = "localhost",
        port: int = 18666,
        non_blocking: bool = False,
        connect_iterations: int = 40,
    ) -> None:
        self.uri = f"tcpip://{ip}:{port}"
        self._handle = BasicStream(
            uri=self.uri,
            agent="S",
            sendBufferSize=DETECTION_ROWS * DETECTION_COLS * 8,
            nonBlocking=non_blocking,
            reshapeOrder="F",
        )
        self._connect(connect_iterations)

    def _connect(self, iterations: int) -> None:
        timeout = Timeout(seconds=0, nanoseconds=100_000)
        for _ in range(iterations):
            if self._handle.connected:
                return
            self._handle.checkConnection(timeout=timeout)
            if self._handle.connected:
                return

    @property
    def connected(self) -> bool:
        return bool(self._handle.connected)

    def send(self, packet: np.ndarray) -> bool:
        """Send one packet. Returns False if not connected."""
        if not self._handle.connected:
            self._connect(1)
            return False
        self._handle.send(packet)
        return True

    def terminate(self) -> None:
        self._handle.terminate()

    def __enter__(self) -> "DetectionStreamServer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.terminate()


@dataclass
class RoadRuleConfig:
    """Thresholds/state timings for road-rule behavior."""

    stop_sign_trigger_m: float = 0.65
    stop_sign_hold_s: float = 2.5
    stop_sign_cooldown_s: float = 6.0
    person_stop_m: float = 1.25
    person_resume_m: float = 1.45


@dataclass
class RoadRuleStatus:
    """Current rule state reported by the state machine."""

    speed_gain: float
    stop_state: str
    stop_sign_distance_m: float
    person_distance_m: float
    person_blocking: bool


class RoadRuleStateMachine:
    """Rule enforcement for stop signs and pedestrians."""

    def __init__(self, config: RoadRuleConfig | None = None) -> None:
        self.cfg = config or RoadRuleConfig()
        self._stop_state = "armed"
        self._stop_state_until = 0.0
        self._person_blocking = False

    def update(
        self,
        stop_sign_row: np.ndarray,
        person_row: np.ndarray,
        now_s: float | None = None,
    ) -> RoadRuleStatus:
        now = time.time() if now_s is None else now_s

        stop_sign_distance = closest_detection_distance(stop_sign_row)
        person_distance = closest_detection_distance(person_row)

        if self._person_blocking:
            self._person_blocking = person_distance < self.cfg.person_resume_m
        else:
            self._person_blocking = person_distance < self.cfg.person_stop_m

        if self._stop_state == "cooldown" and now >= self._stop_state_until:
            self._stop_state = "armed"

        if self._stop_state == "armed" and stop_sign_distance < self.cfg.stop_sign_trigger_m:
            self._stop_state = "stopping"
            self._stop_state_until = now + self.cfg.stop_sign_hold_s

        if self._stop_state == "stopping" and now >= self._stop_state_until:
            self._stop_state = "cooldown"
            self._stop_state_until = now + self.cfg.stop_sign_cooldown_s

        stop_blocking = self._stop_state == "stopping"
        speed_gain = 0.0 if (self._person_blocking or stop_blocking) else 1.0

        return RoadRuleStatus(
            speed_gain=speed_gain,
            stop_state=self._stop_state,
            stop_sign_distance_m=stop_sign_distance,
            person_distance_m=person_distance,
            person_blocking=self._person_blocking,
        )
