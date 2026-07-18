"""
Optional rerun.io visualization for camera frames and CV detections.

Enabled by default:
    .venv/bin/python3 -m mcp_robot.server

    RERUN_ENABLED=   (empty) — disable rerun logging entirely
    RERUN_MODE=spawn (default) — launches the rerun desktop viewer
    RERUN_MODE=serve            — serves a gRPC + web viewer (browser on :9090)

Silently no-ops if rerun-sdk is not installed or RERUN_ENABLED is explicitly disabled.
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
            rrb.Vertical(
                rrb.Spatial2DView(name="Pi Camera", origin="camera/rpi"),
                rrb.Spatial2DView(name="SimpleIPCamera", origin="camera/simpleipcamera"),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="Annotated — Pi Camera", origin="action/pi_camera"),
                rrb.Spatial2DView(name="Annotated — External Camera", origin="action/external_camera"),
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
        recording_id = f"lego_robot_{int(time.time())}"
        if config.RERUN_CONNECT:
            rr.init("lego_robot", recording_id=recording_id)
            rr.connect_grpc(config.RERUN_ADDR)
        elif config.RERUN_MODE == "serve":
            rr.init("lego_robot", recording_id=recording_id)
            uri = rr.serve_grpc()
            rr.serve_web_viewer(connect_to=uri)
        else:
            rr.init("lego_robot", recording_id=recording_id)
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


def _decode_bgr(b64: str):
    """Decode a base64 JPEG into an OpenCV BGR array."""
    import cv2
    import numpy as np
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _caption_bgr(bgr, reason: str):
    """Burn a short caption into the bottom-left corner of a BGR array.

    reason: ~2-3 word explanation of why this frame is being shown (e.g.
            "State check", "Navigation plan"). No-op if empty.
    """
    if not reason or bgr is None:
        return bgr
    import cv2
    out = bgr.copy()
    h = out.shape[0]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), _ = cv2.getTextSize(reason, font, scale, thickness)
    x, y = 8, h - 12
    cv2.rectangle(out, (x - 5, y - th - 7), (x + tw + 5, y + 7), (0, 0, 0), cv2.FILLED)
    cv2.putText(out, reason, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _log_captioned(entity: str, b64: str, reason: str) -> None:
    """Decode, caption, and log one base64 JPEG frame to entity as RGB."""
    import cv2
    bgr = _decode_bgr(b64)
    if bgr is None:
        return
    bgr = _caption_bgr(bgr, reason)
    rr = _rr()
    rr.log(entity, rr.Image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def _log_camera(frame_b64: str, ts: float | None = None, entity: str = "camera/rpi") -> None:
    """Log one frame to the given camera entity with a wall-clock timestamp."""
    rr = _rr()
    rr.set_time("time", timestamp=ts if ts is not None else time.time())
    rr.log(entity, rr.Image(_b64_to_numpy(frame_b64)))


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


def log_simpleipcamera_frame(frame_b64: str, timestamp: float) -> None:
    """Log a SimpleIPCamera frame."""
    if not _ensure_init():
        return
    _log_camera(frame_b64, timestamp, entity="camera/simpleipcamera")


def log_annotated_images(
    pi_b64: str | None = None,
    external_b64: str | None = None,
    reason: str = "",
) -> None:
    """Log annotated action images to the Rerun viewer, routed by camera source.

    pi_b64:       Pi-camera-derived annotated frame -> top row ("Annotated —
                   Pi Camera"), mirroring the raw "Pi Camera" row on the left.
    external_b64: SimpleIPCamera-derived annotated frame (including nav
                   overlay/C-space renders, which are always external-camera
                   based) -> bottom row ("Annotated — External Camera"),
                   mirroring the raw "SimpleIPCamera" row on the left.
    reason:       ~2-3 word caption burned onto the bottom-left corner of
                   each provided frame explaining why it's shown (e.g.
                   "State check", "Navigation plan", "Evaluating execution").
    """
    if not _ensure_init():
        return
    rr = _rr()
    rr.set_time("time", timestamp=time.time())
    if pi_b64:
        _log_captioned("action/pi_camera", pi_b64, reason)
    if external_b64:
        _log_captioned("action/external_camera", external_b64, reason)


def log_nav_tracking(bgr: "np.ndarray", ts: float | None = None, reason: str = "") -> None:
    """Stream a nav-tracking frame to Rerun's external-camera annotated row.

    bgr: OpenCV BGR numpy array (the draw_tracking_overlay output) — always
         SimpleIPCamera-derived, so it belongs on the bottom ("external
         camera") row alongside other navigation annotations.
    """
    if not _ensure_init():
        return
    import cv2 as _cv2
    rr = _rr()
    rr.set_time("time", timestamp=ts if ts is not None else time.time())
    rgb = _cv2.cvtColor(_caption_bgr(bgr, reason), _cv2.COLOR_BGR2RGB)
    rr.log("action/external_camera", rr.Image(rgb))


def flush() -> None:
    """Flush and close the rerun stream. Call before a short-lived script exits."""
    rr = _rr()
    if rr is not None and _initialized:
        rr.disconnect()
