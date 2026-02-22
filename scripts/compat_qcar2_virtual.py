#!/usr/bin/env python3
"""Compatibility helpers for PAL QCar imports in containerized virtual sessions."""

import builtins
import io
import json
import os

from compat_getlogin import patch_os_getlogin

_QCAR2_VIRTUAL_CONFIG = {
    "cartype": 2,
    "carname": "qcar2",
    "lidarurl": "serial-cpu://localhost:1?baud='256000',word='8',parity='none',stop='1',flow='none',dsr='on'",
    "WHEEL_RADIUS": 0.066 / 2,
    "WHEEL_BASE": 0.256,
    "PIN_TO_SPUR_RATIO": (13.0 * 19.0) / (70.0 * 37.0),
    "WRITE_PWM_CHANNELS": [-1],
    "WRITE_OTHER_CHANNELS": [1000, 11000],
    "WRITE_DIGITAL_CHANNELS": [17, 18, 25, 26, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24],
    "writePWMBuffer": 1,
    "writeDigitalBuffer": 16,
    "writeOtherBuffer": 2,
    "READ_ANALOG_CHANNELS": [4, 2],
    "READ_ENCODER_CHANNELS": [0],
    "READ_OTHER_CHANNELS": [3000, 3001, 3002, 4000, 4001, 4002, 14000],
    "readAnalogBuffer": 2,
    "readEncoderBuffer": 1,
    "readOtherBuffer": 7,
    "csiRight": 0,
    "csiBack": 1,
    "csiLeft": 3,
    "csiFront": 2,
    "lidarToGps": "qcar2LidarToGPS.rt-linux_qcar2",
    "captureScan": "qcar2CaptureScan.rt-linux_qcar2",
}


def patch_qcar2_virtual_config():
    """Force PAL QCar configuration to virtual QCar2 without user input or file writes."""

    if getattr(builtins, "_qcar2_virtual_cfg_patched", False):
        return

    patch_os_getlogin()

    config_json = json.dumps(_QCAR2_VIRTUAL_CONFIG)
    real_open = builtins.open
    real_input = builtins.input

    def _safe_open(file, mode="r", *args, **kwargs):
        try:
            path = os.fspath(file)
        except TypeError:
            return real_open(file, mode, *args, **kwargs)

        if isinstance(path, (str, bytes)):
            normalized = os.fsdecode(path).replace("\\", "/")
            if normalized.endswith("/pal/products/qcar_config.json"):
                if any(flag in mode for flag in ("w", "a", "x", "+")):
                    # Discard writes: keep configuration in memory only.
                    return io.StringIO()
                return io.StringIO(config_json)

        return real_open(file, mode, *args, **kwargs)

    def _safe_input(prompt=""):
        if "virtual QCar1 or QCar2" in str(prompt):
            return "2"
        return real_input(prompt)

    builtins.open = _safe_open
    builtins.input = _safe_input
    builtins._qcar2_virtual_cfg_patched = True
