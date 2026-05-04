"""
MCP server entry point for the Lego robot.

Exposes the following tools to MCP clients (e.g. Claude Code):

  Motor primitives
  ─────────────────
  get_robot_state        All motor positions + a live camera snapshot
  get_motor_positions    All motor positions (no camera)
  move_motor             Move a single motor port by N degrees

  Wheel driving
  ─────────────
  drive                  left_speed, right_speed, duration_s (raw — directions uncalibrated)

  Arm & gripper
  ─────────────
  move_arm               Move arm up or down
  control_gripper        Open or close the gripper

  High-level actions
  ──────────────────
  put                    Open gripper + raise arm

  Camera
  ──────
  get_front_camera_image      Capture one still from Pi Camera (front/robot-eye view)
  get_external_camera_image   Capture one still from DroidCam (third-person view)
  capture_front_video_clip    Capture N-second clip from Pi Camera
  capture_external_video_clip Capture N-second clip from DroidCam

Run with:
    python3 -m mcp_robot.server
"""
from __future__ import annotations

import atexit
import base64
import logging
import os
import threading
import time

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

import mcp_robot.camera as cam_mod
import mcp_robot.robot  as robot_mod
from mcp_robot import config, viz, vision

log = logging.getLogger(__name__)
_stop = threading.Event()

# ── initialization tracker ────────────────────────────────────────────────────

_INIT_COMPONENTS = ["motors", "picamera", "droidcam"]
_init_status: dict[str, str] = {c: "pending" for c in _INIT_COMPONENTS}
_init_lock = threading.Lock()


def _log_init_progress(component: str, status: str) -> None:
    with _init_lock:
        _init_status[component] = status
        done    = [c for c in _INIT_COMPONENTS if _init_status[c] == "done"]
        failed  = [c for c in _INIT_COMPONENTS if _init_status[c] == "failed"]
        pending = [c for c in _INIT_COMPONENTS if _init_status[c] == "pending"]
        total   = len(_INIT_COMPONENTS)
        finished = len(done) + len(failed)
        pct = int(finished / total * 100)
        log.info(
            "Initialization %d%% (%d/%d) — done: %s, in-progress: %s, failed: %s",
            pct, finished, total, done, pending, failed,
        )

mcp = FastMCP(
    "lego-robot",
    instructions=(
        "Control a 4-motor Lego robot via BuildHat on a Raspberry Pi. "
        "Motors: left_wheel (A), right_wheel (B), gripper (C), arm (D). "
        "Always call get_robot_state before planning a sequence of actions. "
        "Motor-action tools (move_motor, drive, move_arm, control_gripper, "
        "put) automatically record a video of the motion and return a "
        "vision-model `change_description` summarising what happened — "
        "you do NOT need to call capture_image afterwards to verify them. "
        "Use get_front_camera_image / get_external_camera_image / "
        "capture_front_video_clip / capture_external_video_clip / get_robot_state "
        "when you explicitly need to see the scene. "
        "Stop and report to the user if a motor or camera tool raises an error."
    ),
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ok(data: dict) -> dict:
    return {"ok": True, **data}


def _err(msg: str) -> dict:
    return {"ok": False, "error": msg}


def _image_content(frame_b64: str) -> ImageContent:
    return ImageContent(type="image", data=frame_b64, mimeType="image/jpeg")


def _capture_pair(tag: str = "frame") -> tuple[list[tuple[str, str]], list[str | None]]:
    """Snap one frame from each available camera. Returns (frames, paths)."""
    frames: list[tuple[str, str]] = []
    paths: list[str | None] = []
    try:
        pi = cam_mod.capture_still()
        b64 = pi["frame"]
        frames.append(("pi_camera", b64))
        paths.append(cam_mod._save_snapshot(b64, f"{tag}_pi_camera"))
    except Exception as exc:
        log.warning("Pi camera capture failed during action wrap: %s", exc)
    try:
        droid = cam_mod.capture_droidcam_still()
        b64 = droid["frame"]
        frames.append(("droidcam", b64))
        paths.append(cam_mod._save_snapshot(b64, f"{tag}_droidcam"))
    except Exception as exc:
        log.debug("DroidCam unavailable during action wrap: %s", exc)
    return frames, paths


_ACTION_VIDEO_FPS = 3.0  # frames per second collected during action execution


def _capture_droidcam_background(
    frames: list,
    stop_event: threading.Event,
    fps: float = _ACTION_VIDEO_FPS,
) -> None:
    """Background thread: poll DroidCam and append (ts, b64) pairs to frames."""
    try:
        import cv2
        cap = cv2.VideoCapture(config.DROIDCAM_URL)
        if not cap.isOpened():
            log.debug("Background DroidCam capture: could not open stream")
            return
        interval = 1.0 / fps
        try:
            while not stop_event.is_set():
                t0 = time.time()
                ok, frame = cap.read()
                if ok:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    b64 = base64.b64encode(buf.tobytes()).decode()
                    frames.append((time.time(), b64))
                slack = interval - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)
        finally:
            cap.release()
    except Exception as exc:
        log.debug("Background DroidCam capture failed: %s", exc)


def _with_change_analysis(action_desc: str, expected: str, action_fn) -> dict:
    """
    Record a video of the action, then ask the vision model whether the
    expected outcome was achieved.

    Strategy:
    - Pi camera: slice frames from the streaming cache (populated by stream_live).
    - DroidCam: if streaming cache is active use it; otherwise spin up a
      background cv2 capture thread for the duration of the action.
    - Fallback: if neither camera yields frames, capture before/after stills.

    On action error, returns _err(...) and skips vision.
    On vision failure, the action result is returned without change_description.
    """
    t_start = time.time()

    # Start background DroidCam capture only when its cache is empty (no existing stream)
    droid_bg_frames: list[tuple[float, str]] = []
    stop_event = threading.Event()
    bg_thread: threading.Thread | None = None
    if cam_mod._droidcam_cache.latest() is None:
        bg_thread = threading.Thread(
            target=_capture_droidcam_background,
            args=(droid_bg_frames, stop_event),
            daemon=True,
        )
        bg_thread.start()

    try:
        result = action_fn()
    except Exception as exc:
        stop_event.set()
        if bg_thread:
            bg_thread.join(timeout=2)
        return _err(str(exc))

    time.sleep(config.POST_ACTION_SETTLE)
    stop_event.set()
    if bg_thread:
        bg_thread.join(timeout=2)

    # ── collect video frames in chronological order ───────────────────────────
    video: list[tuple[float, str, str]] = []  # (ts, camera_label, b64)

    pi_clip = cam_mod._pi_cache.clip_since(t_start, _ACTION_VIDEO_FPS)
    if pi_clip:
        for f in pi_clip:
            video.append((f["ts"], "pi_camera", f["frame"]))

    if droid_bg_frames:
        for ts, b64 in droid_bg_frames:
            video.append((ts, "droidcam", b64))
    else:
        droid_clip = cam_mod._droidcam_cache.clip_since(t_start, _ACTION_VIDEO_FPS)
        if droid_clip:
            for f in droid_clip:
                video.append((f["ts"], "droidcam", f["frame"]))

    video.sort(key=lambda x: x[0])
    labeled = [(label, b64) for _, label, b64 in video]

    frame_paths: list[str | None] = [None] * len(video)
    if video and config.SNAPSHOT_DIR:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_start))
        folder = os.path.join(config.SNAPSHOT_DIR, f"action_video_{ts}")
        try:
            os.makedirs(folder, exist_ok=True)
            for i, (_, label, b64) in enumerate(video):
                path = os.path.join(folder, f"{i:03d}_{label}.jpg")
                with open(path, "wb") as fh:
                    fh.write(base64.b64decode(b64))
                frame_paths[i] = path
            log.info("Action video (%d frames) saved to: %s", len(video), folder)
        except Exception as exc:
            log.warning("Failed to save action video: %s", exc)

    out = _ok(result)
    if labeled:
        description = vision.describe_action_video(action_desc, expected, labeled, frame_paths)
    else:
        # No streaming, DroidCam unreachable — fall back to before/after stills
        before, before_paths = _capture_pair("before")
        after, after_paths = _capture_pair("after")
        description = vision.describe_change(
            action_desc, expected, before, after, before_paths, after_paths
        )
    if description:
        out["change_description"] = description
    return out


# ── motor primitives ──────────────────────────────────────────────────────────

@mcp.tool()
def get_motor_positions() -> dict:
    """Return current position (degrees) for all four motor ports."""
    try:
        return _ok(robot_mod.get_all_positions())
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def move_motor(port: str, degrees: int, speed: int = 50) -> dict:
    """
    Move a single motor port by the given number of degrees.

    Captures before/after images from both cameras and returns a
    Gemini-generated `change_description` alongside motor positions.

    Args:
        port:    BuildHat port — "A", "B", "C", or "D".
        degrees: Positive = one direction, negative = opposite.
                 Use small values (e.g. 30–90) to start with.
        speed:   Motor speed, 1–100.
    """
    if port.upper() not in ("A", "B", "C", "D"):
        return _err(f"Invalid port {port!r}. Must be A, B, C or D.")
    if not (1 <= abs(speed) <= 100):
        return _err("speed must be between 1 and 100.")
    p = port.upper()
    role = {
        config.PORT_LEFT_WHEEL:  "left wheel turns (may translate or pivot the robot)",
        config.PORT_RIGHT_WHEEL: "right wheel turns (may translate or pivot the robot)",
        config.PORT_ARM:         "arm moves (positive=down, negative=up)",
        config.PORT_GRIPPER:     "gripper jaws move (positive=close, negative=open)",
    }.get(p, "the connected motor rotates")
    expected = f"motor on port {p} rotates by ~{degrees}°; visually: {role}"
    return _with_change_analysis(
        f"move_motor port={p} degrees={degrees} speed={speed}",
        expected,
        lambda: robot_mod.move_motor(p, degrees, speed),
    )


# ── wheel driving ─────────────────────────────────────────────────────────────

@mcp.tool()
def drive(
    left_speed: int,
    right_speed: int,
    duration_s: float = 1.0,
) -> dict:
    """
    Drive the robot wheels directly. Captures before/after images and returns a
    Gemini-generated `change_description` alongside motor positions.

    Args:
        left_speed:  Speed for the left wheel, -100 to 100. The sign convention
                     (which value moves the robot forward vs backward) must be
                     determined empirically — it has not been calibrated yet.
        right_speed: Speed for the right wheel, -100 to 100.
        duration_s:  How long to run (seconds). Pass 0 to stop both wheels.
    """
    desc = (
        "stop wheels"
        if duration_s == 0
        else f"drive left={left_speed} right={right_speed} for {duration_s}s"
    )
    expected = "wheels spin as commanded; observe droidcam for resulting robot motion"
    return _with_change_analysis(
        desc, expected, lambda: robot_mod.drive(left_speed, right_speed, duration_s),
    )


# ── arm ───────────────────────────────────────────────────────────────────────

@mcp.tool()
def move_arm(degrees: int, speed: int = 30) -> dict:
    """
    Move the robot arm by the given number of degrees. Captures before/after
    images and returns a Gemini-generated `change_description`.

    Args:
        degrees: How far to move. Positive = down, negative = up.
                 Start with values like ±30–90 and adjust based on results.
        speed:   Motor speed, 1–100.
    """
    direction = "down" if degrees > 0 else "up" if degrees < 0 else "no-op"
    expected = (
        f"arm moves {direction} by ~{abs(degrees)}° — visible in droidcam (arm angle "
        f"changes); pi_camera view may tilt as the arm pose shifts; wheels and gripper unchanged"
    )
    return _with_change_analysis(
        f"move arm by {degrees}° (positive=down, negative=up) at speed {speed}",
        expected,
        lambda: robot_mod.move_arm(degrees, speed),
    )


# ── gripper ───────────────────────────────────────────────────────────────────

@mcp.tool()
def control_gripper(action: str, speed: int = 25) -> dict:
    """
    Open or close the gripper. Captures before/after images and returns a
    Gemini-generated `change_description`.

    Args:
        action: "open" or "close".
        speed:  Motor speed, 1–100.
    """
    expected = (
        "gripper jaws open (visibly wider gap between fingers); robot pose and arm unchanged"
        if action == "open"
        else "gripper jaws close (gap narrows; if an object is between them, it is now grasped); robot pose and arm unchanged"
    )
    return _with_change_analysis(
        f"{action} gripper at speed {speed}",
        expected,
        lambda: robot_mod.control_gripper(action, speed),
    )


# ── compound actions ──────────────────────────────────────────────────────────


@mcp.tool()
def put() -> dict:
    """
    High-level PUT: open gripper then raise arm. Captures before/after
    images and returns a Gemini-generated `change_description` confirming
    whether the object was released.
    """
    return _with_change_analysis(
        "put (open gripper + raise arm)",
        "gripper jaws open (releasing any held object so it sits on the surface "
        "in front of the robot), then arm raises — in the AFTER frames the gripper "
        "should be open and the arm should be in its raised position",
        robot_mod.put,
    )


# ── camera ────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_front_camera_image() -> list[ImageContent | TextContent]:
    """
    Capture a single still frame from the Pi Camera (front/robot-eye view).
    Returns the image so you can inspect what the robot sees directly ahead.
    """
    try:
        result = cam_mod.capture_still()
        path_info = f" — saved to {result['path']}" if result.get("path") else ""
        return [
            _image_content(result["frame"]),
            TextContent(
                type="text",
                text=f"Pi Camera — {result['width']}×{result['height']} JPEG ({result['bytes']} bytes){path_info}",
            ),
        ]
    except Exception as exc:
        return [TextContent(type="text", text=f"ERROR: {exc}")]


@mcp.tool()
def get_external_camera_image() -> list[ImageContent | TextContent]:
    """
    Capture a single still frame from the DroidCam (third-person/overhead view).
    Useful for observing the robot's position and surroundings from outside.
    """
    try:
        result = cam_mod.capture_droidcam_still()
        path_info = f" — saved to {result['path']}" if result.get("path") else ""
        return [
            _image_content(result["frame"]),
            TextContent(type="text", text=f"DroidCam — external/third-person view{path_info}"),
        ]
    except Exception as exc:
        return [TextContent(type="text", text=f"ERROR: {exc}")]


@mcp.tool()
def capture_front_video_clip(
    duration_s: float = 2.0,
    fps: float = 2.0,
) -> list[ImageContent | TextContent]:
    """
    Capture a short clip from the Pi Camera (front/robot-eye view).

    Args:
        duration_s: Clip length in seconds (1–10 recommended).
        fps:        Frames per second (1–5 recommended for SSH bandwidth).
    """
    try:
        result = cam_mod.capture_clip(duration_s, fps)
        content: list[ImageContent | TextContent] = [
            TextContent(
                type="text",
                text=f"Pi Camera — {result['count']} frames at {fps:.1f} fps ({duration_s}s)",
            )
        ]
        for frame_b64 in result["frames"]:
            content.append(_image_content(frame_b64))
        vqa = vision.describe_clip("pi_camera", result["frames"], result.get("paths"))
        if vqa:
            content.append(TextContent(type="text", text=f"Clip VQA (Qwen):\n{vqa}"))
        return content
    except Exception as exc:
        return [TextContent(type="text", text=f"ERROR: {exc}")]


@mcp.tool()
def capture_external_video_clip(
    duration_s: float = 2.0,
    fps: float = 2.0,
) -> list[ImageContent | TextContent]:
    """
    Capture a short clip from the DroidCam (third-person/overhead view).

    Args:
        duration_s: Clip length in seconds (1–10 recommended).
        fps:        Frames per second (1–5 recommended).
    """
    try:
        result = cam_mod.capture_droidcam_clip(duration_s, fps)
        content: list[ImageContent | TextContent] = [
            TextContent(
                type="text",
                text=f"DroidCam — {result['count']} frames at {fps:.1f} fps ({duration_s}s)",
            )
        ]
        for frame_b64 in result["frames"]:
            content.append(_image_content(frame_b64))
        vqa = vision.describe_clip("droidcam", result["frames"], result.get("paths"))
        if vqa:
            content.append(TextContent(type="text", text=f"Clip VQA (Qwen):\n{vqa}"))
        return content
    except Exception as exc:
        return [TextContent(type="text", text=f"ERROR: {exc}")]


@mcp.tool()
def get_robot_state() -> list[ImageContent | TextContent]:
    """
    One-shot state snapshot: all motor positions + live frames from both
    cameras (Pi Camera = front view; DroidCam = wider third-person view).
    Call this before planning any sequence of actions.
    """
    try:
        positions = robot_mod.get_all_positions()
        pi_frame = cam_mod.capture_still()
        summary = (
            f"Motor positions — "
            f"left_wheel: {positions['left_wheel']}°, "
            f"right_wheel: {positions['right_wheel']}°, "
            f"arm: {positions['arm']}°, "
            f"gripper: {positions['gripper']}°"
        )
        content: list[ImageContent | TextContent] = [
            TextContent(type="text", text="Pi Camera (front view):"),
            _image_content(pi_frame["frame"]),
        ]
        try:
            droid_frame = cam_mod.capture_droidcam_still()
            content.append(TextContent(type="text", text="DroidCam (third-person view):"))
            content.append(_image_content(droid_frame["frame"]))
        except Exception as exc:
            content.append(TextContent(type="text", text=f"DroidCam unavailable: {exc}"))
        content.append(TextContent(type="text", text=summary))
        return content
    except Exception as exc:
        return [TextContent(type="text", text=f"ERROR: {exc}")]


# ── background streaming ──────────────────────────────────────────────────────

def _run_pi_camera() -> None:
    reported = [False]

    def _on_frame(frame: str, ts: float) -> None:
        if not reported[0]:
            reported[0] = True
            _log_init_progress("picamera", "done")
        viz.log_frame(frame, ts)

    backoff = 1.0
    while not _stop.is_set():
        try:
            cam_mod.stream_live(fps=5.0, stop_event=_stop, on_frame=_on_frame)
        except Exception as exc:
            if not reported[0]:
                _log_init_progress("picamera", "failed")
            log.warning("Pi Camera stream ended: %s", exc)
        if not _stop.is_set():
            log.info("Pi Camera reconnecting in %.0fs...", backoff)
            _stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)
        else:
            break


def _run_droidcam() -> None:
    reported = [False]

    def _on_frame(frame: str, ts: float) -> None:
        if not reported[0]:
            reported[0] = True
            _log_init_progress("droidcam", "done")
        viz.log_droidcam_frame(frame, ts)

    backoff = 1.0
    while not _stop.is_set():
        try:
            cam_mod.stream_droidcam(stop_event=_stop, on_frame=_on_frame)
        except Exception as exc:
            if not reported[0]:
                _log_init_progress("droidcam", "failed")
            log.warning("DroidCam stream ended: %s", exc)
        if not _stop.is_set():
            log.info("DroidCam reconnecting in %.0fs...", backoff)
            _stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)
        else:
            break


def _start_background_streams() -> None:
    log.info("Initialization 0%% (0/%d) — done: [], in-progress: %s, failed: []",
             len(_INIT_COMPONENTS), _INIT_COMPONENTS)
    try:
        robot_mod.get_all_positions()
        _log_init_progress("motors", "done")
    except Exception as exc:
        log.warning("Motor init failed (%s) — continuing without motor data.", exc)
        _log_init_progress("motors", "failed")

    for target, name in [
        (_run_pi_camera,  "pi-camera"),
        (_run_droidcam,   "droidcam"),
    ]:
        threading.Thread(target=target, name=name, daemon=True).start()

    atexit.register(_shutdown)


def _shutdown() -> None:
    _stop.set()
    viz.flush()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    log_file = config.LOG_FILE
    # basicConfig is a no-op if any handler already exists (FastMCP sets one up
    # during __init__ at module level), so explicitly add the FileHandler instead.
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        log.info("Logging to %s", log_file)
    if config.RERUN_ENABLED:
        _start_background_streams()
    mcp.run()


if __name__ == "__main__":
    main()
