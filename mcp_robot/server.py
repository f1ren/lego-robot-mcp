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
  drive                  left_speed, right_speed, duration_s (positive = forward for both wheels)

  Arm & gripper
  ─────────────
  move_arm               Move arm up or down (downward moves end with a 17° raise)
  lower_arm              Lower arm fully to ground then raise 17° for wheel clearance
  control_gripper        Open or close the gripper (open ends with 17° close-back to release wheel pressure)

  High-level actions
  ──────────────────
  put                    Open gripper + raise arm

  Camera
  ──────
  get_front_camera_image      Capture one still from Pi Camera (front/robot-eye view)
  get_external_camera_image   Capture one still from DroidCam (third-person view)
  capture_front_video_clip    Capture N-second clip from Pi Camera
  capture_external_video_clip Capture N-second clip from DroidCam

  Vision / localization
  ─────────────────────
  locate_object               VLM-based localization of arbitrary objects (Gemini Flash)

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

import cv2
import numpy as np

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

import mcp_robot.camera as cam_mod
import mcp_robot.robot  as robot_mod
from mcp_robot import config, heading, viz, vision
from mcp_robot import grasp_readiness as grasp_mod
from mcp_robot import navigation as nav_mod

log = logging.getLogger(__name__)
_stop = threading.Event()

# Motor encoder positions are relative to wherever the motors were when the
# server started (or last power-cycled). Reporting them on the very first
# get_robot_state call would give the AI meaningless baseline numbers that
# look authoritative but carry no positional information. We suppress them
# on the first call and only emit them from the second call onward, when the
# AI already has a prior reading to delta-compare against.
_state_call_count = 0

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
        "Motor-action tools (move_motor, drive, move_arm, lower_arm, control_gripper, "
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


# Image size caps for frames returned to the MCP client (i.e. shown to Claude).
# Claude Code truncates MCP tool output at ~25 000 tokens.  A 640×480 JPEG at
# quality 85 is ~80–120 KB, which base64-encodes to ~110–160 KB (~30–46 K
# tokens) — well over the limit.  Keep all client-visible images small:
#   • _MAX_PIXELS  = 172 800 ≈ 480×360 — used for single-frame image tools
#   • _CLIP_MAX_PIXELS = 76 800 ≈ 320×240 — used for multi-frame video clips
#   • _STATE_MAX_PIXELS = 76 800 ≈ 320×240 — compact thumbnail for get_robot_state
# VQA calls (Gemini / Ollama) are made server-side and receive the full-res
# frames from disk — this cap only affects what arrives in the Claude context.
_MAX_PIXELS       = 172_800   # ≈480×360 — ~15–30 KB JPEG → ~6–12 K tokens
_CLIP_MAX_PIXELS  =  76_800   # ≈320×240 — per-frame cap for video clips
_STATE_MAX_PIXELS =  76_800   # ≈320×240 — small thumbnail for get_robot_state


def _scale_jpeg_b64(frame_b64: str, max_pixels: int = _MAX_PIXELS, quality: int = 70) -> str:
    """Scale and re-encode a base64 JPEG to ≤ max_pixels at the given quality.

    Always re-encodes (never passes through raw bytes) so the quality reduction
    is applied even when the frame is already within the pixel budget.
    """
    raw = base64.b64decode(frame_b64)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return frame_b64
    h, w = img.shape[:2]
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return frame_b64
    return base64.b64encode(buf.tobytes()).decode()


def _image_content(frame_b64: str) -> ImageContent:
    """Single-frame image at ≈480×360, quality 70 (~15–30 KB, ~5–10 K tokens)."""
    return ImageContent(type="image", data=_scale_jpeg_b64(frame_b64), mimeType="image/jpeg")


def _clip_image_content(frame_b64: str) -> ImageContent:
    """Per-frame image for video clips at ≈320×240, quality 65 (~8–15 KB, ~3–5 K tokens)."""
    data = _scale_jpeg_b64(frame_b64, max_pixels=_CLIP_MAX_PIXELS, quality=65)
    return ImageContent(type="image", data=data, mimeType="image/jpeg")


def _thumbnail_image_content(frame_b64: str) -> ImageContent:
    """Compact thumbnail (320×240 max, quality 65) for state snapshots."""
    data = _scale_jpeg_b64(frame_b64, max_pixels=_STATE_MAX_PIXELS, quality=65)
    return ImageContent(type="image", data=data, mimeType="image/jpeg")



_ACTION_VIDEO_FPS = 5.0  # frames per second collected during action execution


def _capture_droidcam_background(
    frames: list,
    stop_event: threading.Event,
    fps: float = _ACTION_VIDEO_FPS,
) -> None:
    """Background thread: poll DroidCam and append (ts, b64) pairs to frames.

    Frames are stored raw (no heading arrow) so the VLM sees clean before/after
    comparisons without the arrow creating spurious motion detections.
    cap.read() blocks until the next frame arrives from DroidCam's MJPEG stream,
    naturally capping at DroidCam's native rate; we add a sleep only when our
    target fps is lower than what the camera delivers.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(config.DROIDCAM_URL)
        if not cap.isOpened():
            log.debug("Background DroidCam capture: could not open stream")
            return
        target_fps = config.DROIDCAM_CAPTURE_FPS
        interval = 1.0 / target_fps
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


def _apply_optical_flow_per_camera(
    annotated: list[tuple[float, str, str]],
    raw: list[tuple[float, str, str]],
) -> list[tuple[float, str, str]]:
    """Add optical flow red arrows to annotated frames, processed per-camera.

    Flow is computed from raw frames (no heading-arrow overlay) so the static
    arrow pixels don't produce spurious flow vectors.
    """
    cam_indices: dict[str, list[int]] = {}
    for i, (_, label, _) in enumerate(annotated):
        cam_indices.setdefault(label, []).append(i)

    cam_raw_b64: dict[str, list[str]] = {}
    for _, label, b64 in raw:
        cam_raw_b64.setdefault(label, []).append(b64)

    result = list(annotated)
    for label, indices in cam_indices.items():
        ann_b64s = [annotated[i][2] for i in indices]
        flow_b64s = heading.annotate_flow_sequence_jpeg_b64(ann_b64s, cam_raw_b64.get(label))
        for idx, new_b64 in zip(indices, flow_b64s):
            ts, lbl, _ = result[idx]
            result[idx] = (ts, lbl, new_b64)
    return result


def _with_change_analysis(
    action_desc: str,
    expected: str,
    action_fn,
    context: str = "",
    annotate: bool = True,
    vqa_cameras: set[str] | None = None,
) -> dict:
    """
    Record a video of the action, then ask the vision model whether the
    expected outcome was achieved.

    Strategy:
    - Pi camera: slice frames from the streaming cache (populated by stream_live).
    - DroidCam: if streaming cache is active use it; otherwise spin up a
      background cv2 capture thread for the duration of the action.
    - Fallback: if neither camera yields frames, capture before/after stills.

    annotate: whether to overlay the heading arrow on DroidCam frames sent to
              the VQA model. Pass False for arm/gripper actions so the arrow
              does not cover the arm or gripper and confuse the model about
              their state. Pass True (default) for drive/turn actions where
              heading information is needed for evaluation.
    vqa_cameras: set of camera labels to include in the VQA call (e.g.
              {"droidcam"}). None (default) means all cameras. Frames from
              excluded cameras are still saved to disk but not sent to VQA.

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
    # raw_video holds unannotated frames (for motion gate only).
    # annotated_video holds arrow-overlaid frames (for VQA and saved clips).
    raw_video: list[tuple[float, str, str]] = []
    annotated_video: list[tuple[float, str, str]] = []

    def _maybe_annotate(b64: str) -> str:
        return heading.annotate_jpeg_b64(b64) if annotate else b64

    if droid_bg_frames:
        for ts, b64 in droid_bg_frames:
            raw_video.append((ts, "droidcam", b64))
            annotated_video.append((ts, "droidcam", _maybe_annotate(b64)))
    else:
        droid_clip = cam_mod._droidcam_cache.clip_since(t_start, config.DROIDCAM_CAPTURE_FPS)
        if droid_clip:
            for f in droid_clip:
                raw_video.append((f["ts"], "droidcam", f["frame"]))
                annotated_video.append((f["ts"], "droidcam", _maybe_annotate(f["frame"])))

    pi_clip = cam_mod._pi_cache.clip_since(t_start, config.PICAMERA_CAPTURE_FPS)
    if pi_clip:
        for f in pi_clip:
            raw_video.append((f["ts"], "pi_camera", f["frame"]))
            annotated_video.append((f["ts"], "pi_camera", f["frame"]))

    annotated_video.sort(key=lambda x: x[0])
    raw_video.sort(key=lambda x: x[0])
    annotated_video = _apply_optical_flow_per_camera(annotated_video, raw_video)
    labeled = [(label, b64) for _, label, b64 in annotated_video]
    raw_labeled = [(label, b64) for _, label, b64 in raw_video]

    frame_paths: list[str | None] = [None] * len(annotated_video)
    if annotated_video and config.SNAPSHOT_DIR:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_start))
        folder = os.path.join(config.SNAPSHOT_DIR, f"action_video_{ts}")
        annotated_folder = os.path.join(folder, "annotated")
        try:
            os.makedirs(folder, exist_ok=True)
            os.makedirs(annotated_folder, exist_ok=True)
            cam_counters: dict[str, int] = {}
            for i, ((_, label, raw_b64), (_, _, ann_b64)) in enumerate(zip(raw_video, annotated_video)):
                idx = cam_counters.get(label, 0)
                cam_counters[label] = idx + 1
                fname = f"{label}_{idx:03d}.jpg"
                # raw frame → main folder (read by compile_video)
                with open(os.path.join(folder, fname), "wb") as fh:
                    fh.write(base64.b64decode(raw_b64))
                # annotated frame → subfolder (for VQA path logging only)
                ann_path = os.path.join(annotated_folder, fname)
                with open(ann_path, "wb") as fh:
                    fh.write(base64.b64decode(ann_b64))
                frame_paths[i] = ann_path
            log.info("Action video (%d frames) saved to: %s (annotated → %s/annotated/)",
                     len(annotated_video), folder, folder)
        except Exception as exc:
            log.warning("Failed to save action video: %s", exc)

    if not labeled:
        raise RuntimeError(
            f"No video frames captured during action {action_desc!r} — "
            "at least one camera (DroidCam or Pi Camera) must be streaming."
        )

    out = _ok(result)
    if vqa_cameras is not None:
        vqa_indices = [i for i, (lbl, _) in enumerate(labeled) if lbl in vqa_cameras]
        vqa_labeled = [labeled[i] for i in vqa_indices]
        vqa_raw = [raw_labeled[i] for i in vqa_indices]
        vqa_paths = [frame_paths[i] for i in vqa_indices]
    else:
        vqa_labeled, vqa_raw, vqa_paths = labeled, raw_labeled, frame_paths

    # Build per-camera stacks from all annotated frames (not just VQA subset).
    cam_frames: dict[str, list[str]] = {}
    for lbl, b64 in labeled:
        cam_frames.setdefault(lbl, []).append(b64)
    cam_stacks = [
        vision.stack_frames(cam_b64s) if len(cam_b64s) > 1 else cam_b64s[0]
        for cam_b64s in cam_frames.values()
    ]
    viz.log_annotated_images(
        cam_stacks[0] if len(cam_stacks) > 0 else None,
        cam_stacks[1] if len(cam_stacks) > 1 else None,
    )

    description = vision.describe_action_video(
        action_desc, expected, vqa_labeled, vqa_paths, context=context,
        raw_labeled_frames=vqa_raw,
    )
    if description:
        out["change_description"] = description
    return out


# ── motor primitives ──────────────────────────────────────────────────────────

@mcp.tool()
def get_motor_positions() -> dict:
    """Return current position (degrees) for all four motor ports."""
    log.info("[TOOL] get_motor_positions")
    try:
        return _ok(robot_mod.get_all_positions())
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def move_motor(port: str, degrees: int, speed: int = 20, expected: str = "", context: str = "") -> dict:
    """
    Move a single motor port by the given number of degrees.

    Captures before/after images from both cameras and returns a
    Gemini-generated `change_description` alongside motor positions.

    Args:
        port:     BuildHat port — "A", "B", "C", or "D".
        degrees:  Positive = one direction, negative = opposite.
                  Use small values (e.g. 30–90) to start with.
        speed:    Motor speed, 15–20.
        expected: Short, precise description of what should physically happen
                  (e.g. "arm rotates down 45°"). Defaults to a technical summary.
        context:  Why this action is being taken and hints for evaluation
                  (e.g. "lowering arm to ball height; last attempt overshot by 20°").
    """
    log.info("[TOOL] move_motor port=%r degrees=%r speed=%r", port, degrees, speed)
    if port.upper() not in ("A", "B", "C", "D"):
        return _err(f"Invalid port {port!r}. Must be A, B, C or D.")
    speed_max = config.ARM_SPEED_MAX if port.upper() == config.PORT_ARM else config.SPEED_MAX
    if not (config.SPEED_MIN <= abs(speed) <= speed_max):
        return _err(f"speed must be between {config.SPEED_MIN} and {speed_max} (abs).")
    p = port.upper()
    role = {
        config.PORT_LEFT_WHEEL:  "left wheel turns (may translate or pivot the robot)",
        config.PORT_RIGHT_WHEEL: "right wheel turns (may translate or pivot the robot)",
        config.PORT_ARM:         "arm moves (positive=down, negative=up)",
        config.PORT_GRIPPER:     "gripper jaws move (positive=close, negative=open)",
    }.get(p, "the connected motor rotates")
    expected_str = expected if expected else f"motor on port {p} rotates by ~{degrees}°; visually: {role}"
    # Wheel ports (A, B) need the heading arrow for direction evaluation.
    # Arm (D) and gripper (C) ports must NOT have the arrow — it masks their state.
    wheel_ports = {config.PORT_LEFT_WHEEL, config.PORT_RIGHT_WHEEL}
    is_wheel = p in wheel_ports
    return _with_change_analysis(
        f"move_motor port={p} degrees={degrees} speed={speed}",
        expected_str,
        lambda: robot_mod.move_motor(p, degrees, speed),
        context=context,
        annotate=is_wheel,
        vqa_cameras={"droidcam"} if is_wheel else None,
    )


# ── wheel driving ─────────────────────────────────────────────────────────────

@mcp.tool()
def drive(
    left_speed: int,
    right_speed: int,
    duration_s: float = 1.0,
    expected: str = "",
    context: str = "",
) -> dict:
    """
    Drive the robot wheels directly. Captures before/after images and returns a
    Gemini-generated `change_description` alongside motor positions.

    Args:
        left_speed:  Speed for the left wheel, -20 to -15, 0, or 15 to 20. Positive = forward.
        right_speed: Speed for the right wheel, -20 to -15, 0, or 15 to 20. Positive = forward.
                     To rotate/turn, use inverse values, left_speed = -right_speed.
        duration_s:  How long to run (seconds). Pass 0 to stop both wheels.
        expected:    Short, precise description of the expected motion
                     (e.g. "robot moves forward ~20 cm").
        context:     Why this action is being taken and hints for evaluation
                     (e.g. "approaching the ball; previous attempt turned clockwise instead").
    """
    log.info("[TOOL] drive left_speed=%r right_speed=%r duration_s=%r", left_speed, right_speed, duration_s)
    for name, val in (("left_speed", left_speed), ("right_speed", right_speed)):
        if val != 0 and not (config.SPEED_MIN <= abs(val) <= config.SPEED_MAX):
            return _err(f"{name} must be 0 or between {config.SPEED_MIN} and {config.SPEED_MAX} (abs).")
    desc = (
        "stop wheels"
        if duration_s == 0
        else f"drive left={left_speed} right={right_speed} for {duration_s}s"
    )
    expected_str = expected if expected else "robot moves or pivots; observe droidcam for direction and distance"
    return _with_change_analysis(
        desc, expected_str, lambda: robot_mod.drive(left_speed, right_speed, duration_s),
        context=context,
        vqa_cameras={"droidcam"},
    )


@mcp.tool()
def turn(
    body_degrees: float,
    speed: int = 20,
    expected: str = "",
    context: str = "",
) -> dict:
    """
    Rotate the robot body in place by an exact number of degrees.

    Use only for fine-grained heading corrections when you are already close to the
    target. For navigating toward an object, call navigate_to() instead — it handles
    turning AND obstacle avoidance automatically.

    Conversion: encoder_deg = abs(body_degrees) * (TRACK_WIDTH_MM / WHEEL_DIAMETER_MM).
    Defaults: track_width=123 mm, wheel_diameter=56 mm → ratio ≈ 2.2.
    Override via TURN_ENCODER_DEG_PER_BODY_DEG env var if calibration drifts.

    Args:
        body_degrees: Signed rotation in degrees viewed from above.
                      Positive = clockwise (CW), negative = counter-clockwise (CCW).
        speed:        Wheel speed, 15–20.
        expected:     Short description of the expected outcome
                      (e.g. "robot rotates ~45° CW to face the cup").
        context:      Why this action is being taken and hints for evaluation.
    """
    log.info("[TOOL] turn body_degrees=%r speed=%r", body_degrees, speed)
    if not (config.SPEED_MIN <= abs(speed) <= config.SPEED_MAX):
        return _err(f"speed must be between {config.SPEED_MIN} and {config.SPEED_MAX} (abs).")
    direction = "CW" if body_degrees >= 0 else "CCW"
    encoder_deg = int(abs(body_degrees) * config.TURN_ENCODER_DEG_PER_BODY_DEG)
    desc = f"turn {body_degrees}° ({direction}), encoder_deg={encoder_deg}"
    expected_str = expected if expected else (
        f"robot rotates ~{abs(body_degrees):.0f}° {direction} in place"
    )
    return _with_change_analysis(
        desc, expected_str,
        lambda: robot_mod.turn(body_degrees, speed),
        context=context,
        vqa_cameras={"droidcam"},
    )


# ── arm ───────────────────────────────────────────────────────────────────────

@mcp.tool()
def move_arm(degrees: int, speed: int = 15, expected: str = "", context: str = "") -> dict:
    """
    Move the robot arm by the given number of degrees. Captures before/after
    images and returns a Gemini-generated `change_description`.

    Args:
        degrees:  How far to move. Positive = down, negative = up.
                  Start with values like ±30–90 and adjust based on results.
        speed:    Motor speed, 15 (max 15 to prevent arm jitter).
        expected: Short, precise description of the expected outcome
                  (e.g. "arm moves down ~45°, tip reaches ball height").
        context:  Why this action is being taken and hints for evaluation
                  (e.g. "positioning arm to grasp ball; last attempt stopped too high").
    """
    log.info("[TOOL] move_arm degrees=%r speed=%r", degrees, speed)
    if not (config.SPEED_MIN <= abs(speed) <= config.ARM_SPEED_MAX):
        return _err(f"arm speed must be between {config.SPEED_MIN} and {config.ARM_SPEED_MAX} (abs).")
    direction = "down" if degrees > 0 else "up" if degrees < 0 else "no-op"
    suffix = "; then raises 17° to clear gripper from ground" if degrees > 0 else ""
    expected_str = expected if expected else (
        f"arm moves {direction} by ~{abs(degrees)}°{suffix} — visible in 3rd party cam (arm angle "
        f"changes); front camera may show arm entering or leaving frame; wheels and gripper unchanged"
    )
    return _with_change_analysis(
        f"move arm by {degrees}° (positive=down, negative=up) at speed {speed}",
        expected_str,
        lambda: robot_mod.move_arm(degrees, speed),
        context=context,
        annotate=False,  # arrow masks arm position — suppress for arm actions
    )


@mcp.tool()
def lower_arm(speed: int = 15, expected: str = "", context: str = "") -> dict:
    """
    Lower the robot arm fully to ground level, then raise it 17° to keep the
    gripper clear of the floor and maximise wheel normal force. Captures
    before/after images and returns a Gemini-generated `change_description`.

    Args:
        speed:    Motor speed, 15 (max 15 to prevent arm jitter).
        expected: Short, precise description of the expected outcome.
        context:  Why this action is being taken and hints for evaluation.
    """
    log.info("[TOOL] lower_arm speed=%r", speed)
    if not (config.SPEED_MIN <= abs(speed) <= config.ARM_SPEED_MAX):
        return _err(f"arm speed must be between {config.SPEED_MIN} and {config.ARM_SPEED_MAX} (abs).")
    expected_str = expected if expected else (
        f"arm lowers fully to ground level (~{config.ARM_DOWN_DEG}°), then raises 17° "
        "so the gripper just clears the floor; final arm position is slightly above ground"
    )
    return _with_change_analysis(
        f"lower arm to ground level then raise 17° at speed {speed}",
        expected_str,
        lambda: robot_mod.lower_arm(speed),
        context=context,
        annotate=False,
    )


# ── gripper ───────────────────────────────────────────────────────────────────

@mcp.tool()
def control_gripper(
    action: str,
    speed: int = 20,
    expected: str = "",
    context: str = "",
    target_class_yolo: str = "cup",
    target_class_free_text: str = "",
) -> dict:
    """
    Open or close the gripper. Captures before/after images and returns a
    Gemini-generated `change_description`.

    Args:
        action:                 "open" or "close".
        speed:                  Motor speed, 15–20.
        expected:               Short, precise description of the expected outcome
                                (e.g. "gripper closes around the ball").
        context:                Why this action is being taken and hints for
                                evaluation (e.g. "grasping paper ball; previous
                                close attempt slipped off").
        target_class_yolo:      YOLO class for the grasp-readiness gate
                                (e.g. "cup", "ball", "any"). Default: "cup".
        target_class_free_text: Free-text description for Gemini Flash fallback
                                when YOLO finds nothing (e.g. "light switch").
    """
    log.info("[TOOL] control_gripper action=%r speed=%r yolo=%r free_text=%r",
             action, speed, target_class_yolo, target_class_free_text)
    if not (config.SPEED_MIN <= abs(speed) <= config.SPEED_MAX):
        return _err(f"speed must be between {config.SPEED_MIN} and {config.SPEED_MAX} (abs).")

    if action == "open":
        # Opening almost always succeeds — skip VQA to avoid the expense.
        try:
            return {"ok": True, "change_description": "Skipped description, because gripper 'open' almost always works", **robot_mod.control_gripper(action, speed)}
        except Exception as exc:
            return _err(str(exc))

    # Gate: check grasp readiness before closing.
    frame_result = cam_mod.capture_droidcam_still(annotate=False)
    import numpy as _np, cv2 as _cv2
    raw = base64.b64decode(frame_result["frame"])
    arr = _np.frombuffer(raw, dtype=_np.uint8)
    bgr = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
    if bgr is None:
        return _err("Grasp readiness gate: could not decode external camera frame.")
    readiness = grasp_mod.check_grasp_readiness(
        bgr,
        target_class_yolo=target_class_yolo,
        target_class_free_text=target_class_free_text,
    )
    if not readiness.ready:
        log.warning("[TOOL] control_gripper blocked by grasp readiness gate: %s", readiness.reason)
        return _err(f"Grasp readiness check failed — gripper NOT closed.\n{readiness.to_text()}")

    # Closing: verify that any object between the fingers is now grasped.
    expected_str = expected if expected else (
        "gripper jaws close (gap narrows; if an object is between them, it is now grasped); robot pose and arm unchanged"
    )
    return _with_change_analysis(
        f"close gripper at speed {speed}",
        expected_str,
        lambda: robot_mod.control_gripper(action, speed),
        context=context,
        annotate=False,  # arrow masks gripper state — suppress for gripper actions
    )


# ── compound actions ──────────────────────────────────────────────────────────


@mcp.tool()
def put() -> dict:
    """
    High-level PUT: open gripper then raise arm. Captures before/after
    images and returns a Gemini-generated `change_description` confirming
    whether the object was released.
    """
    log.info("[TOOL] put")
    return _with_change_analysis(
        "put (open gripper + raise arm)",
        "gripper jaws open (releasing any held object so it sits on the surface "
        "in front of the robot), then arm raises — in the AFTER frames the gripper "
        "should be open and the arm should be in its raised position",
        robot_mod.put,
        annotate=False,  # arrow masks gripper/arm state — suppress for put action
    )


# ── grasp readiness ───────────────────────────────────────────────────────────

@mcp.tool()
def check_grasp_readiness(
    target_class_yolo: str = "cup",
    target_class_free_text: str = "",
) -> list[TextContent]:
    """
    CV-based grasp readiness check using the external (DroidCam) camera.

    Captures a live frame and verifies two conditions required before closing
    the gripper:
      1. The target object is touching the robot's front body.
      2. The green forward-arrow passes well over the object's center of mass.

    Args:
        target_class_yolo:      YOLO class to look for. Supported values:
                                  "cup"    — cup / bowl / bottle / vase  (default)
                                  "ball"   — sports ball / orange / apple
                                  "bottle" — bottle / cup / vase
                                  "any"    — most-forward object of any class
                                Pass "" to skip YOLO entirely.
        target_class_free_text: Free-text description sent to Gemini Flash when
                                YOLO finds nothing (e.g. "light switch").
                                Leave empty to disable the VLM path.

    Returns a verdict, a human-readable reason, and — when not ready — an
    actionable next step (drive closer, adjust heading, etc.).
    """
    log.info("[TOOL] check_grasp_readiness yolo=%r free_text=%r", target_class_yolo, target_class_free_text)
    try:
        frame_result = cam_mod.capture_droidcam_still(annotate=False)
        raw = base64.b64decode(frame_result["frame"])
        import numpy as _np, cv2 as _cv2
        arr = _np.frombuffer(raw, dtype=_np.uint8)
        bgr = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        if bgr is None:
            return [TextContent(type="text", text="ERROR: could not decode external camera frame.")]
        result = grasp_mod.check_grasp_readiness(
            bgr,
            target_class_yolo=target_class_yolo,
            target_class_free_text=target_class_free_text,
        )
        return [TextContent(type="text", text=result.to_text())]
    except Exception as exc:
        log.warning("[TOOL] check_grasp_readiness error: %s", exc)
        return [TextContent(type="text", text=f"ERROR: {exc}")]


# ── navigation ────────────────────────────────────────────────────────────────

def _nav_track_motor(
    base_overlay: np.ndarray,
    stop_event: threading.Event,
) -> None:
    """Background thread for navigate_to: stream DroidCam frames with CV
    robot-position tracking overlaid on the planned path to Rerun.

    Called while motors are running; replaces VQA for step verification.
    """
    trail: list[tuple[int, int]] = []

    def _on_frame(bgr: np.ndarray, ts: float) -> None:
        robot_px = nav_mod.detect_robot_px(bgr)
        if robot_px is not None:
            trail.append(robot_px)
            if len(trail) > 40:
                trail.pop(0)
        tracking = nav_mod.draw_tracking_overlay(base_overlay, trail)
        viz.log_nav_tracking(tracking, ts)

    cam_mod.stream_droidcam_bgr(_on_frame, stop_event)


@mcp.tool()
def navigate_to(
    target_class_yolo: str,
    target_class_free_text: str,
    max_steps: int = 6,
) -> list[ImageContent | TextContent]:
    """
    Navigate the robot toward a target object using CV-based obstacle avoidance.

    At every step the tool:
      1. Captures an external (DroidCam) frame and detects robot + target.
      2. Builds a pixel-resolution obstacle map (floor segmentation + yellow-body
         exclusion) and a coarse navigable grid.
      3. Runs A* from the robot's grid cell to the target's grid cell.
      4. Saves step_NN_raw.jpg, step_NN_obstacle_mask.jpg, and
         step_NN_nav_overlay.jpg to SNAPSHOT_DIR/navigate_to_<ts>/ for
         visual verification and unit-test fixtures.
      5. Executes the next turn + drive command while a background thread
         continuously tracks the robot's yellow body via CV and streams the
         live position overlaid on the planned path to Rerun (navigation/tracking).
      6. Repeats until the robot is at the target, the path is blocked, or
         max_steps is reached.

    Returns a stacked composite of all per-step overlay frames so you can
    see the full trajectory at a glance, plus a text log.

    Args:
        target_class_yolo:      YOLO class for the target (e.g. "cup", "ball",
                                "bottle", "any"). Pass "" to skip YOLO.
                                REQUIRED — pass "" explicitly to skip YOLO.
        target_class_free_text: Free-text description for Gemini Flash fallback
                                when YOLO finds nothing (e.g. "light switch").
                                REQUIRED — pass "" explicitly to disable the VLM
                                fallback. Both target_class_yolo and
                                target_class_free_text must always be supplied
                                so the target is unambiguous; do not rely on
                                defaults.
        max_steps:              Maximum navigation steps (default 6).
    """
    log.info("[TOOL] navigate_to yolo=%r free_text=%r max_steps=%r",
             target_class_yolo, target_class_free_text, max_steps)

    ts = time.strftime("%Y%m%d_%H%M%S")
    nav_dir = os.path.join(config.SNAPSHOT_DIR, f"navigate_to_{ts}") if config.SNAPSHOT_DIR else ""

    key_frames_b64: list[str] = []
    step_logs: list[str] = []
    outcome = "max_steps_reached"

    obs_map: nav_mod.ObstacleMap | None = None
    plan: nav_mod.NavPlan | None = None

    try:
        for step in range(max_steps):
            parts: list[str] = [f"=== Step {step + 1}/{max_steps} ==="]

            # ── 1. Capture external frame ─────────────────────────────────
            try:
                frame_result = cam_mod.capture_droidcam_still(annotate=False)
            except Exception as exc:
                parts.append(f"Camera capture failed: {exc}")
                step_logs.append("\n".join(parts))
                outcome = "camera_error"
                break

            raw_bytes = base64.b64decode(frame_result["frame"])
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                parts.append("Could not decode camera frame")
                step_logs.append("\n".join(parts))
                outcome = "camera_error"
                break

            # ── 2. Detect robot heading (cheap, every step) ───────────────
            h_result = heading.detect_heading(bgr)
            if h_result is None:
                parts.append("WARNING: robot yellow body not detected — heading unknown")
            else:
                parts.append(f"Robot at {h_result.body_center}, "
                             f"forward={tuple(round(v, 2) for v in h_result.forward)}")

            if step == 0:
                # ── 3. Target detection (YOLO + VLM) — first step only ────
                target_obj = None
                if target_class_yolo:
                    try:
                        objects = grasp_mod._yolo_detect(bgr, target_class=target_class_yolo)
                        if objects:
                            target_obj = (
                                grasp_mod._pick_target(objects, h_result)
                                if h_result is not None
                                else max(objects, key=lambda o: o.confidence)
                            )
                    except Exception as exc:
                        parts.append(f"YOLO error: {exc}")

                if target_obj is None and target_class_free_text:
                    target_obj = grasp_mod._vlm_detect(bgr, target_class_free_text)

                if target_obj is None:
                    parts.append(f"Target not detected "
                                 f"({target_class_yolo or target_class_free_text})")
                else:
                    parts.append(f"Target '{target_obj.class_name}' "
                                 f"at {target_obj.center} conf={target_obj.confidence:.0%}")
                    if target_obj.note:
                        parts.append(f"VLM note: {target_obj.note}")

                # ── 4. Full obstacle detection — first step only ──────────
                obs_map = nav_mod.detect_obstacles(bgr, h_result, target_obj)
                free_frac = (obs_map.free_mask > 0).mean()
                parts.append(f"Obstacle map: {free_frac:.0%} free space")

                # ── 5. Path planning ──────────────────────────────────────
                plan = nav_mod.plan_path(obs_map)
                parts.append(f"Path: {plan.reason}")

                # ── 6. Save debug images ──────────────────────────────────
                if nav_dir:
                    try:
                        nav_mod.save_debug_images(bgr, obs_map, plan, nav_dir, step)
                    except Exception as exc:
                        log.warning("Failed to save nav debug images: %s", exc)

            else:
                # ── 3–5. Subsequent steps: track position, replan if needed ──
                robot_px = nav_mod.detect_robot_px(bgr)
                if robot_px is not None:
                    deviation = nav_mod.dist_to_path(robot_px, plan.path_px)
                    nav_mod.update_robot_position(obs_map, robot_px)
                    if deviation > obs_map.robot_radius_px:
                        parts.append(
                            f"Deviation {deviation:.0f}px > {obs_map.robot_radius_px:.0f}px "
                            f"— replanning on existing C-space"
                        )
                        plan = nav_mod.plan_path(obs_map)
                    else:
                        parts.append(f"On path (deviation {deviation:.0f}px)")
                    parts.append(f"Path: {plan.reason}")
                else:
                    parts.append("WARNING: robot not detected — keeping previous plan")

            # ── 7. Collect key frame (nav overlay + cspace) ───────────────
            overlay_bgr = nav_mod.draw_nav_overlay(bgr, obs_map, plan, step + 1)
            ok_enc, overlay_buf = cv2.imencode(
                ".jpg", overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            overlay_b64: str | None = None
            if ok_enc:
                overlay_b64 = base64.b64encode(overlay_buf.tobytes()).decode()
                key_frames_b64.append(overlay_b64)

            cspace_bgr = nav_mod.build_cspace_bgr(bgr, obs_map, plan)
            ok_cs, cs_buf = cv2.imencode(".jpg", cspace_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
            cspace_b64 = base64.b64encode(cs_buf.tobytes()).decode() if ok_cs else None
            viz.log_annotated_images(cspace_b64, overlay_b64)

            # ── 8. Termination checks ─────────────────────────────────────
            if nav_mod.near_target(obs_map):
                parts.append("NEAR TARGET — within C-space buffer distance, "
                             "stopping to avoid collision — navigation complete")
                step_logs.append("\n".join(parts))
                outcome = "success"
                break

            if not plan.reachable:
                parts.append(f"PATH BLOCKED — {plan.reason}")
                step_logs.append("\n".join(parts))
                outcome = "path_blocked"
                break

            # ── 9. Execute next step ──────────────────────────────────────
            turn_deg, drive_deg = nav_mod.commands_for_step(obs_map, plan, h_result)
            parts.append(f"Commands: turn={turn_deg:+.0f}°, drive={drive_deg:.0f}° (wheel)")

            _track_stop = threading.Event()
            _track_thread = threading.Thread(
                target=_nav_track_motor,
                args=(overlay_bgr, _track_stop),
                daemon=True,
            )
            _track_thread.start()
            try:
                if abs(turn_deg) > 8:
                    direction = "CW" if turn_deg > 0 else "CCW"
                    log.info("[navigate_to] step %d — turn %+.0f° %s",
                             step + 1, turn_deg, direction)
                    robot_mod.turn(float(turn_deg), config.NAV_TURN_SPEED)
                log.info("[navigate_to] step %d — drive %.0f° (encoder) forward",
                         step + 1, drive_deg)
                robot_mod.drive_degrees(int(round(drive_deg)), config.SPEED_MAX, config.SPEED_MAX)
            finally:
                _track_stop.set()
                _track_thread.join(timeout=3.0)

            step_logs.append("\n".join(parts))

        else:
            step_logs.append(
                f"Reached max_steps ({max_steps}) without arriving at target"
            )

    except Exception as exc:
        log.warning("[TOOL] navigate_to error: %s", exc, exc_info=True)
        step_logs.append(f"ERROR: {exc}")
        outcome = "error"

    # ── Final report ───────────────────────────────────────────────────────
    content: list[ImageContent | TextContent] = []

    if key_frames_b64:
        try:
            stacked_b64 = (
                vision.stack_frames(key_frames_b64)
                if len(key_frames_b64) > 1
                else key_frames_b64[0]
            )
            content.append(_image_content(stacked_b64))
        except Exception as exc:
            log.warning("navigate_to: could not stack key frames: %s", exc)

    outcome_text = {
        "success":           "Navigation successful — robot is at the target.",
        "path_blocked":      "Navigation failed — no obstacle-free path found.",
        "camera_error":      "Navigation aborted — camera error.",
        "error":             "Navigation aborted — unexpected error.",
        "max_steps_reached": f"Navigation incomplete — max_steps ({max_steps}) reached without reaching target.",
    }.get(outcome, outcome)

    log_text = "\n\n".join(step_logs) if step_logs else "(no steps executed)"
    if nav_dir:
        log_text += f"\n\nDebug images: {nav_dir}"
    content.append(
        TextContent(type="text", text=f"Navigate-to outcome: {outcome_text}\n\n{log_text}")
    )
    return content


# ── VLM object localization ───────────────────────────────────────────────────

@mcp.tool()
def locate_object(description: str) -> list[ImageContent | TextContent]:
    """
    Locate an arbitrary object using Gemini Flash vision localization.

    Use this for objects that YOLO cannot detect — anything outside the COCO-80
    class set, such as "light switch", "door handle", "power outlet",
    "red cable connector", "white box on the shelf", etc.

    Captures the external (DroidCam) camera, asks Gemini Flash to find the
    object described by *description*, and returns:
      • An annotated frame with the heading arrow, object bounding box, and
        angle label overlaid.
      • The angle from the robot's current forward direction to the object
        (positive = CW, negative = CCW, viewed from above).
      • A short note from the VLM describing what it found.

    Pass the returned angle directly to `turn(body_degrees=<angle>)` to face
    the object before driving toward it.

    Args:
        description: Free-text description of the object to find, e.g.
                     "light switch", "door handle", "yellow power strip".
    """
    log.info("[TOOL] locate_object description=%r", description)
    try:
        # Capture external camera (unannotated — VLM should see the raw scene)
        frame_result = cam_mod.capture_droidcam_still(annotate=False)
        raw = base64.b64decode(frame_result["frame"])
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return [TextContent(type="text", text="ERROR: could not decode external camera frame.")]

        # Ask Gemini Flash to locate the object
        vlm_result = vision.locate_object_vlm(bgr, description)
        if vlm_result is None:
            # Return an unannotated (heading-arrow only) frame + helpful message
            annotated_bgr = heading.annotate_bgr(bgr)
            ok, buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buf.tobytes()).decode() if ok else frame_result["frame"]
            return [
                _image_content(frame_b64),
                TextContent(
                    type="text",
                    text=(
                        f"Object not found: '{description}' was not detected in the external "
                        "camera frame. Check that the object is visible and try again."
                    ),
                ),
            ]

        (x1, y1, x2, y2), confidence, note = vlm_result
        obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Compute heading angle to object
        h_result = heading.detect_heading(bgr)
        angle_deg: float | None = None
        angle_text = ""
        if h_result is not None:
            angle_deg = heading.compute_heading_to_object_angle(h_result, obj_center)
            rot_dir = "CW" if angle_deg > 0 else "CCW"
            angle_text = (
                f"Object-to-heading angle: {abs(angle_deg):.0f}° {rot_dir} "
                f"(positive=CW, negative=CCW, viewed from above). "
                f"Pass this as body_degrees to turn() to face the object."
            )

        # Annotate frame: heading arrow + object bbox + angle line
        annotated_bgr = heading.annotate_bgr(
            bgr,
            obj_center=obj_center,
            obj_bbox=(x1, y1, x2, y2),
        )
        ok, buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            frame_b64 = frame_result["frame"]
        else:
            frame_b64 = base64.b64encode(buf.tobytes()).decode()

        content: list[ImageContent | TextContent] = [
            _image_content(frame_b64),
            TextContent(
                type="text",
                text=(
                    f"Located '{description}' via Gemini Flash VLM "
                    f"(conf={confidence:.0%}) at pixel bbox [{x1},{y1},{x2},{y2}], "
                    f"center={obj_center}."
                ),
            ),
        ]
        if note:
            content.append(TextContent(type="text", text=f"VLM note: {note}"))
        if angle_text:
            content.append(TextContent(type="text", text=angle_text))
        elif h_result is None:
            content.append(TextContent(
                type="text",
                text="Heading not detected — robot yellow body not visible; angle cannot be computed.",
            ))
        return content

    except Exception as exc:
        log.warning("[TOOL] locate_object error: %s", exc, exc_info=True)
        return [TextContent(type="text", text=f"ERROR: {exc}")]


# ── camera ────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_front_camera_image() -> list[ImageContent | TextContent]:
    """
    Capture a single still frame from the Pi Camera (front/robot-eye view).
    Returns the image so you can inspect what the robot sees directly ahead.
    """
    log.info("[TOOL] get_front_camera_image")
    try:
        result = cam_mod.capture_still()
        viz.log_annotated_images(result["frame"])
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
    log.info("[TOOL] get_external_camera_image")
    try:
        result = cam_mod.capture_droidcam_still()
        viz.log_annotated_images(result["frame"])
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
    log.info("[TOOL] capture_front_video_clip duration_s=%r fps=%r", duration_s, fps)
    try:
        result = cam_mod.capture_clip(duration_s, fps)
        content: list[ImageContent | TextContent] = [
            TextContent(
                type="text",
                text=f"Pi Camera — {result['count']} frames at {fps:.1f} fps ({duration_s}s)",
            )
        ]
        for frame_b64 in result["frames"]:
            content.append(_clip_image_content(frame_b64))
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
    log.info("[TOOL] capture_external_video_clip duration_s=%r fps=%r", duration_s, fps)
    try:
        result = cam_mod.capture_droidcam_clip(duration_s, fps)
        content: list[ImageContent | TextContent] = [
            TextContent(
                type="text",
                text=f"DroidCam — {result['count']} frames at {fps:.1f} fps ({duration_s}s)",
            )
        ]
        for frame_b64 in result["frames"]:
            content.append(_clip_image_content(frame_b64))
        vqa = vision.describe_clip(
            "droidcam", result["frames"], result.get("paths"),
            raw_frames=result.get("raw_frames"),
        )
        if vqa:
            content.append(TextContent(type="text", text=f"Clip VQA (Qwen):\n{vqa}"))
        return content
    except Exception as exc:
        return [TextContent(type="text", text=f"ERROR: {exc}")]


@mcp.tool()
def get_robot_state(
    target_class_yolo: str = "cup",
    target_class_free_text: str = "",
) -> list[ImageContent | TextContent]:
    """
    One-shot state snapshot: all motor positions + live frames from both
    cameras (Pi Camera = front view; DroidCam = wider third-person view).
    Call this before planning any sequence of actions.

    Args:
        target_class_yolo:      YOLO class to detect and annotate in the external
                                camera frame. Supported values:
                                  "cup"    → cup, bowl, bottle, vase  (default)
                                  "ball"   → sports ball, orange, apple
                                  "bottle" → bottle, cup, vase
                                  "any"    → most-forward object of any class
                                Pass "" to skip YOLO.
        target_class_free_text: Free-text description for Gemini Flash when YOLO
                                finds nothing (e.g. "light switch", "door handle").
                                The detected object's angle from the robot's heading
                                is returned so you know how much to turn.
    """
    global _state_call_count
    log.info("[TOOL] get_robot_state yolo=%r free_text=%r", target_class_yolo, target_class_free_text)
    try:
        _state_call_count += 1
        content: list[ImageContent | TextContent] = []
        try:
            droid_frame = cam_mod.capture_droidcam_still(
                target_class_yolo=target_class_yolo,
                target_class_free_text=target_class_free_text,
            )
            viz.log_annotated_images(droid_frame["frame"])
            content.append(TextContent(type="text", text="Third-person view 320×240 thumbnail:"))
            content.append(_thumbnail_image_content(droid_frame["frame"]))
            angle_deg = droid_frame.get("object_angle_deg")
            vlm_note = droid_frame.get("vlm_note")
            if angle_deg is not None:
                rot_dir = "CW" if angle_deg > 0 else "CCW"
                angle_text = (
                    f"Object-to-heading angle: {abs(angle_deg):.0f}° {rot_dir} "
                    f"(positive=CW, negative=CCW, viewed from above). "
                    f"Use navigate_to(yolo=..., free_text=...) to move toward the object "
                    f"with automatic obstacle avoidance. Do NOT manually plan turn + drive sequences."
                )
                log.info("get_robot_state heading result: %s", angle_text)
                content.append(TextContent(type="text", text=angle_text))
            else:
                log.info("get_robot_state heading result: no object detected (yolo=%r free_text=%r) — angle not computed",
                         target_class_yolo, target_class_free_text)
            if vlm_note:
                content.append(TextContent(type="text", text=f"VLM note: {vlm_note}"))
        except Exception as exc:
            content.append(TextContent(type="text", text=f"Third-person view unavailable: {exc}"))
        if _state_call_count > 1:
            positions = robot_mod.get_all_positions()
            summary = (
                f"Motor positions — "
                f"left_wheel: {positions['left_wheel']}°, "
                f"right_wheel: {positions['right_wheel']}°, "
                f"arm: {positions['arm']}°, "
                f"gripper: {positions['gripper']}°"
            )
            content.append(TextContent(type="text", text=summary))
        return content
    except Exception as exc:
        return [TextContent(type="text", text=f"ERROR: {exc}")]


# ── task video ────────────────────────────────────────────────────────────────

@mcp.tool()
def compile_video(since: str, output_fps: float = config.DROIDCAM_CAPTURE_FPS) -> dict:
    """
    Compile a task video from all action-video folders captured since a given
    timestamp. Only frames where motion was detected (pixel diff vs. neighbour
    above threshold) are included.

    Call this after a sequence of actions to get a single video of the task.

    Args:
        since:      ISO-style datetime string marking the start of the task,
                    e.g. "20260507_143022" (same format as folder names), or a
                    UNIX timestamp as a string (e.g. "1746613200.0").
        output_fps: Playback frame rate of the output video (default: DROIDCAM_CAPTURE_FPS).

    Returns a dict with video_path, total_frames, motion_frames, action_count.
    """
    log.info("[TOOL] compile_video since=%r output_fps=%r", since, output_fps)
    from mcp_robot.video_compiler import compile_task_video
    result = compile_task_video(since, config.SNAPSHOT_DIR, output_fps)
    if not result.ok:
        return _err(result.error)
    if result.video_path is None:
        return _ok({"message": "No motion detected across all frames — video not written",
                    "total_frames": result.total_frames, "motion_frames": 0,
                    "action_count": result.action_count})
    return _ok({"video_path": result.video_path, "total_frames": result.total_frames,
                "motion_frames": result.motion_frames, "action_count": result.action_count,
                "output_fps": output_fps})


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
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        log.info("Logging to %s", log_file)
    if config.RERUN_ENABLED:
        _start_background_streams()
    mcp.run()


if __name__ == "__main__":
    main()
