"""
Optional rerun.io visualization for camera frames and CV detections.

Usage:
    RERUN_ENABLED=1 .venv/bin/python3 -m mcp_robot.server

    RERUN_MODE=spawn  (default) — launches the rerun desktop viewer
    RERUN_MODE=serve            — serves a gRPC + web viewer (browser on :9090)

Silently no-ops if rerun-sdk is not installed or RERUN_ENABLED is unset.
"""
from __future__ import annotations

import base64
import os
import sys
import time
from io import BytesIO

from mcp_robot import config

_initialized = False
_init_failed = False


def _rr():
    try:
        import rerun as rr
        return rr
    except ImportError:
        return None


def _venv_rerun() -> str | None:
    """Return path to the rerun binary bundled with the current venv, if present."""
    candidate = os.path.join(os.path.dirname(sys.executable), "rerun")
    return candidate if os.path.isfile(candidate) else None


def _send_blueprint() -> None:
    import rerun.blueprint as rrb
    rr = _rr()
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Camera", origin="camera"),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Motors", origin="motors"),
                rrb.TextLogView(name="Vision log", origin="vision"),
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)


def _ensure_init() -> bool:
    global _initialized, _init_failed
    if _init_failed:
        return False
    rr = _rr()
    if rr is None or not config.RERUN_ENABLED:
        return False
    if not _initialized:
        if config.RERUN_CONNECT:
            rr.init("lego_robot", recording_id="lego_robot_session")
            rr.connect_grpc(config.RERUN_ADDR)
        elif config.RERUN_MODE == "serve":
            rr.init("lego_robot", recording_id="lego_robot_session")
            uri = rr.serve_grpc()
            rr.serve_web_viewer(connect_to=uri)
        else:
            rr.init("lego_robot", recording_id="lego_robot_session")
            try:
                rr.spawn(executable_path=_venv_rerun())
            except RuntimeError as exc:
                print(f"[viz] rerun viewer not found ({exc}). "
                      "Use RERUN_MODE=serve or run: pip install rerun-sdk")
                _init_failed = True
                return False
        _send_blueprint()
        _initialized = True
    return True


def _b64_to_numpy(b64: str):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))


def _log_camera(frame_b64: str, ts: float | None = None) -> None:
    """Log one frame to the single camera entity with a wall-clock timestamp."""
    rr = _rr()
    rr.set_time("time", timestamp=ts if ts is not None else time.time())
    rr.log("camera", rr.Image(_b64_to_numpy(frame_b64)))


def log_frame(frame_b64: str, timestamp: float) -> None:
    """Log a live-stream frame."""
    if not _ensure_init():
        return
    _log_camera(frame_b64, timestamp)


def log_still(frame_dict: dict) -> None:
    """Log a captured still frame."""
    if not _ensure_init():
        return
    _log_camera(frame_dict["frame"])


def log_clip(clip_dict: dict) -> None:
    """Log a clip's frames at evenly-spaced synthetic timestamps."""
    if not _ensure_init():
        return
    now = time.time()
    count = len(clip_dict["frames"])
    for i, frame_b64 in enumerate(clip_dict["frames"]):
        _log_camera(frame_b64, now - (count - 1 - i) * 0.5)


def log_verify(before_b64: str, after_b64: str, result: dict) -> None:
    """Log before/after frames and the Gemini verdict."""
    if not _ensure_init():
        return
    rr = _rr()
    now = time.time()
    _log_camera(before_b64, now - 1.0)
    _log_camera(after_b64, now)

    success = result.get("success")
    confidence = result.get("confidence", "?")
    explanation = result.get("explanation", "")
    level = rr.TextLogLevel.INFO if success else rr.TextLogLevel.WARN
    rr.set_time("time", timestamp=now)
    rr.log("vision/verify", rr.TextLog(f"[{confidence}] {explanation}", level=level))


_motor_series_logged: set[str] = set()


def log_motor_positions(positions: dict) -> None:
    """Log a {port: degrees} dict as scalar timeseries."""
    if not _ensure_init():
        return
    rr = _rr()
    rr.set_time("time", timestamp=time.time())
    for port, deg in positions.items():
        if isinstance(deg, (int, float)):
            entity = f"motors/{port}"
            if entity not in _motor_series_logged:
                rr.log(entity, rr.SeriesLines(names=[port]), static=True)
                _motor_series_logged.add(entity)
            rr.log(entity, rr.Scalars(float(deg)))


def flush() -> None:
    """Flush and close the rerun stream. Call before a short-lived script exits."""
    rr = _rr()
    if rr is not None and _initialized:
        rr.disconnect()
