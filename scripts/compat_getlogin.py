#!/usr/bin/env python3
"""Compatibility helper for environments where os.getlogin() is unavailable."""

from __future__ import annotations

import getpass
import os


def patch_os_getlogin() -> None:
    """Patch os.getlogin with a safe fallback if needed.

    Quanser HAL/PAL/PIT modules call os.getlogin() at import time.
    In containerized sessions without a login TTY, os.getlogin() raises OSError.
    """

    if getattr(os, "_qcar_safe_getlogin_patched", False):
        return

    original_getlogin = os.getlogin

    def _safe_getlogin() -> str:
        try:
            return original_getlogin()
        except OSError:
            for key in ("LOGNAME", "USER", "LNAME", "USERNAME"):
                value = os.environ.get(key)
                if value:
                    return value
            try:
                return getpass.getuser()
            except Exception:
                return "unknown"

    os.getlogin = _safe_getlogin  # type: ignore[assignment]
    os._qcar_safe_getlogin_patched = True  # type: ignore[attr-defined]

