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
  lift_arm               Close gripper (hold torque), lift arm to home/retracted position, hold, release gripper
  control_gripper        Open or close the gripper (open ends with 17° close-back to release wheel pressure)

  High-level actions
  ──────────────────
  put                    Open gripper + raise arm

  Camera
  ──────
  get_front_camera_image      Capture one still from Pi Camera (front/robot-eye view)
  get_external_camera_image   Capture one still from SimpleIPCamera (third-person view)
  capture_front_video_clip    Capture N-second clip from Pi Camera
  capture_external_video_clip Capture N-second clip from SimpleIPCamera

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

# ── target distance guard ────────────────────────────────────────────────────
# Stored by get_robot_state when a target is detected, cleared by navigate_to.
# drive/turn check this and refuse when the robot is far from the target.
_last_target_distance_px: float | None = None
_last_target_robot_radius_px: float | None = None
_last_target_yolo: str = ""
_last_target_free_text: str = ""

# ── initialization tracker ────────────────────────────────────────────────────

_INIT_COMPONENTS = ["motors", "picamera", "simpleipcamera"]
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


def _target_too_far() -> str | None:
    """If a known target is beyond near-target range, return an error message.

    Returns None when no guard applies (no target, or target is close enough).
    """
    if _last_target_distance_px is None or _last_target_robot_radius_px is None:
        return None
    threshold = (_last_target_robot_radius_px
                 * nav_mod._CSPACE_BUFFER_SCALE
                 * nav_mod._NEAR_TARGET_GRACE)
    if _last_target_distance_px <= threshold:
        return None
    yolo = _last_target_yolo
    free = _last_target_free_text
    return (
        f"ERROR: Target is too far for manual drive/turn "
        f"(distance={_last_target_distance_px:.0f}px, threshold={threshold:.0f}px). "
        f"Use navigate_to(target_class_yolo={yolo!r}, "
        f"target_class_free_text={free!r}) instead."
    )


def _resolve_target(target_class_yolo: str, target_class_free_text: str) -> tuple[str, str]:
    """Fall back to the last target seen by navigate_to()/click_button() when
    both explicit target params are blank. If either is given explicitly, use
    exactly what was given — never mix an explicit field with a stale one
    from a prior, unrelated target.
    """
    if not target_class_yolo and not target_class_free_text:
        return _last_target_yolo, _last_target_free_text
    return target_class_yolo, target_class_free_text


def _measure_target(target_class_yolo: str, target_class_free_text: str) -> tuple[float | None, float | None]:
    """Capture a fresh simpleipcamera still and measure *target*'s angle off the
    robot's forward heading and straight-line distance, converted to mm.

    Updates the _last_target_* globals as a side effect (same fields
    get_robot_state writes), so _target_too_far() reflects this fresh
    reading immediately afterward.

    Returns (angle_deg, distance_mm) — either is None if the target wasn't
    detected; distance_mm alone may be None if the target was seen but the
    robot's own body plate wasn't visible for px->mm calibration.
    """
    global _last_target_distance_px, _last_target_robot_radius_px
    global _last_target_yolo, _last_target_free_text
    frame_result = cam_mod.capture_simpleipcamera_still(
        target_class_yolo=target_class_yolo,
        target_class_free_text=target_class_free_text,
    )
    viz.log_annotated_images(external_b64=frame_result["frame"], reason="Target check")
    _last_target_distance_px = frame_result.get("object_distance_px")
    _last_target_robot_radius_px = frame_result.get("robot_radius_px")
    _last_target_yolo = target_class_yolo
    _last_target_free_text = target_class_free_text
    dist_px = frame_result.get("object_distance_px")
    body_area_px = frame_result.get("robot_body_area_px")
    distance_mm = None
    if dist_px is not None and body_area_px is not None:
        scale = nav_mod.mm_per_px(body_area_px)
        if scale is not None:
            distance_mm = dist_px * scale
    return frame_result.get("object_angle_deg"), distance_mm


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



def _capture_simpleipcamera_background(stop_event: threading.Event) -> None:
    """Background thread: poll SimpleIPCamera and feed frames into _simpleipcamera_cache.

    Frames are stored raw (no heading arrow) so the VLM sees clean before/after
    comparisons without the arrow creating spurious motion detections. Feeding
    _simpleipcamera_cache also lets the SegmentRecorder observe these frames.
    cap.read() blocks until the next frame arrives from SimpleIPCamera's MJPEG stream,
    naturally capping at SimpleIPCamera's native rate; we add a sleep only when our
    target fps is lower than what the camera delivers.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(config.SIMPLEIPCAMERA_URL)
        if not cap.isOpened():
            log.debug("Background SimpleIPCamera capture: could not open stream")
            return
        target_fps = config.SIMPLEIPCAMERA_CAPTURE_FPS
        interval = 1.0 / target_fps
        try:
            while not stop_event.is_set():
                t0 = time.time()
                ok, frame = cap.read()
                if ok:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    b64 = base64.b64encode(buf.tobytes()).decode()
                    cam_mod._simpleipcamera_cache.put(b64, time.time())
                slack = interval - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)
        finally:
            cap.release()
    except Exception as exc:
        log.debug("Background SimpleIPCamera capture failed: %s", exc)


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
    skip_vqa: bool = False,
    sub_observation: str = "",
    sub_action: str = "",
) -> dict:
    """
    Record a video of the action, then ask the vision model whether the
    expected outcome was achieved.

    Strategy:
    - Pi camera: slice frames from the streaming cache (populated by stream_live).
    - SimpleIPCamera: if streaming cache is active use it; otherwise spin up a
      background cv2 capture thread for the duration of the action.
    - Fallback: if neither camera yields frames, capture before/after stills.

    annotate: whether to overlay the heading arrow on SimpleIPCamera frames sent to
              the VQA model. Pass False for arm/gripper actions so the arrow
              does not cover the arm or gripper and confuse the model about
              their state. Pass True (default) for drive/turn actions where
              heading information is needed for evaluation.
    vqa_cameras: set of camera labels to include in the VQA call (e.g.
              {"simpleipcamera"}). None (default) means all cameras.

    On action error, returns _err(...) and skips vision.
    On vision failure, the action result is returned without change_description
    but with a `vqa_error` field explaining what went wrong — the caller must
    not treat a missing change_description as "nothing to report"; it means
    the action's outcome could not be verified and the tool call may be worth
    recalling, or the failure worth investigating in code.

    Recorded motion segments (mcp_robot.recorder.SegmentRecorder) overlapping
    [t_start, t_end] are tagged with this action's tool name and
    change_description.
    """
    t_start = time.time()

    # Start background SimpleIPCamera capture only when its cache is empty (no existing stream)
    stop_event: threading.Event | None = None
    bg_thread: threading.Thread | None = None
    if cam_mod._simpleipcamera_cache.latest() is None:
        stop_event = threading.Event()
        bg_thread = threading.Thread(
            target=_capture_simpleipcamera_background,
            args=(stop_event,),
            daemon=True,
        )
        bg_thread.start()

    try:
        result = action_fn()
    except Exception as exc:
        log.error("[TOOL] %s error: %s", action_desc, exc, exc_info=True)
        if stop_event:
            stop_event.set()
        if bg_thread:
            bg_thread.join(timeout=2)
        return _err(str(exc))

    time.sleep(config.POST_ACTION_SETTLE)
    if stop_event:
        stop_event.set()
    if bg_thread:
        bg_thread.join(timeout=2)

    t_end = time.time()

    # ── collect video frames in chronological order ───────────────────────────
    # raw_video holds unannotated frames (for motion gate only).
    # annotated_video holds arrow-overlaid frames (for VQA).
    raw_video: list[tuple[float, str, str]] = []
    annotated_video: list[tuple[float, str, str]] = []

    def _maybe_annotate(b64: str) -> str:
        return heading.annotate_jpeg_b64(b64) if annotate else b64

    simpleipcam_clip = cam_mod._simpleipcamera_cache.clip_since(t_start, config.SIMPLEIPCAMERA_CAPTURE_FPS)
    if simpleipcam_clip:
        for f in simpleipcam_clip:
            raw_video.append((f["ts"], "simpleipcamera", f["frame"]))
            annotated_video.append((f["ts"], "simpleipcamera", _maybe_annotate(f["frame"])))

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

    if not labeled:
        raise RuntimeError(
            f"No video frames captured during action {action_desc!r} — "
            "at least one camera (SimpleIPCamera or Pi Camera) must be streaming."
        )

    out = _ok(result)
    if vqa_cameras is not None:
        vqa_indices = [i for i, (lbl, _) in enumerate(labeled) if lbl in vqa_cameras]
        vqa_labeled = [labeled[i] for i in vqa_indices]
        vqa_raw = [raw_labeled[i] for i in vqa_indices]
    else:
        vqa_labeled, vqa_raw = labeled, raw_labeled

    # Build per-camera stacks from all annotated frames (not just VQA subset).
    cam_frames: dict[str, list[str]] = {}
    for lbl, b64 in labeled:
        cam_frames.setdefault(lbl, []).append(b64)

    def _stack(cam_b64s: list[str] | None) -> str | None:
        if not cam_b64s:
            return None
        return vision.stack_frames(cam_b64s) if len(cam_b64s) > 1 else cam_b64s[0]

    viz.log_annotated_images(
        pi_b64=_stack(cam_frames.get("pi_camera")),
        external_b64=_stack(cam_frames.get("simpleipcamera")),
        reason="Evaluating execution",
    )

    description = None
    if not skip_vqa:
        try:
            description = vision.describe_action_video(
                action_desc, expected, vqa_labeled, None, context=context,
                raw_labeled_frames=vqa_raw,
            )
        except vision.VQAFailure as exc:
            log.error("[TOOL] %s VQA failed: %s", action_desc, exc)
            out["vqa_error"] = str(exc)
    if description:
        out["change_description"] = description

    from mcp_robot.recorder import get_recorder
    rec = get_recorder()
    meta = {"tool": action_desc, "change_description": description,
            "sub_observation": sub_observation or None,
            "sub_action": sub_action or None}
    for cam in cam_frames:
        rec.tag_range(cam, t_start, t_end, meta)

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
def move_motor(port: str, degrees: int, speed: int = 20, expected: str = "", context: str = "",
               sub_observation: str = "", sub_action: str = "") -> dict:
    """
    Move a single motor port by the given number of degrees.

    Captures before/after images from both cameras and returns a
    Gemini-generated `change_description` alongside motor positions.

    Args:
        port:     BuildHat port — "A", "B", "C", or "D".
        degrees:  Positive = one direction, negative = opposite.
                  Use small values (e.g. 30–90) to start with.
        speed:    Motor speed, 15–20 (port D / arm: 7–15).
        expected: Short, precise description of what should physically happen
                  (e.g. "arm rotates down 45°"). Defaults to a technical summary.
        context:  Why this action is being taken and hints for evaluation
                  (e.g. "lowering arm to ball height; last attempt overshot by 20°").
        sub_observation: ~4-word video subtitle: what was just observed or instructed
                         (e.g. "Cup detected nearby").
        sub_action:      ~4-word video subtitle: what the robot is doing now
                         (e.g. "Lowering arm down").
    """
    log.info("[TOOL] move_motor port=%r degrees=%r speed=%r", port, degrees, speed)
    if port.upper() not in ("A", "B", "C", "D"):
        return _err(f"Invalid port {port!r}. Must be A, B, C or D.")
    is_arm = port.upper() == config.PORT_ARM
    speed_min = config.ARM_SPEED_MIN if is_arm else config.SPEED_MIN
    speed_max = config.ARM_SPEED_MAX if is_arm else config.SPEED_MAX
    if not (speed_min <= abs(speed) <= speed_max):
        return _err(f"speed must be between {speed_min} and {speed_max} (abs).")
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
        vqa_cameras={"simpleipcamera"} if is_wheel else None,
        sub_observation=sub_observation,
        sub_action=sub_action,
    )


# ── wheel driving ─────────────────────────────────────────────────────────────

@mcp.tool()
def drive(
    left_speed: int,
    right_speed: int,
    duration_s: float = 1.0,
    expected: str = "",
    context: str = "",
    sub_observation: str = "",
    sub_action: str = "",
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
        sub_observation: ~4-word video subtitle: what was just observed or instructed
                         (e.g. "Cup is ahead").
        sub_action:      ~4-word video subtitle: what the robot is doing now
                         (e.g. "Driving forward").
    """
    log.info("[TOOL] drive left_speed=%r right_speed=%r duration_s=%r", left_speed, right_speed, duration_s)
    guard = _target_too_far()
    if guard:
        return _err(guard)
    for name, val in (("left_speed", left_speed), ("right_speed", right_speed)):
        if val != 0 and not (config.SPEED_MIN <= abs(val) <= config.SPEED_MAX):
            return _err(f"{name} must be 0 or between {config.SPEED_MIN} and {config.SPEED_MAX} (abs).")
    desc = (
        "stop wheels"
        if duration_s == 0
        else f"drive left={left_speed} right={right_speed} for {duration_s}s"
    )
    expected_str = expected if expected else "robot moves or pivots; observe simpleipcamera for direction and distance"
    return _with_change_analysis(
        desc, expected_str, lambda: robot_mod.drive(left_speed, right_speed, duration_s),
        context=context,
        vqa_cameras={"simpleipcamera"},
        sub_observation=sub_observation,
        sub_action=sub_action,
    )


@mcp.tool()
def turn(
    body_degrees: float,
    speed: int = 20,
    expected: str = "",
    context: str = "",
    sub_observation: str = "",
    sub_action: str = "",
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
        sub_observation: ~4-word video subtitle: what was just observed or instructed
                         (e.g. "Cup is to the left").
        sub_action:      ~4-word video subtitle: what the robot is doing now
                         (e.g. "Turning to face cup").
    """
    log.info("[TOOL] turn body_degrees=%r speed=%r", body_degrees, speed)
    guard = _target_too_far()
    if guard:
        return _err(guard)
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
        vqa_cameras={"simpleipcamera"},
        sub_observation=sub_observation,
        sub_action=sub_action,
    )


@mcp.tool()
def turn_to(
    target_class_yolo: str = "",
    target_class_free_text: str = "",
    tolerance_deg: float = config.TURN_TO_TOLERANCE_DEG,
    speed: int = 20,
    expected: str = "",
    context: str = "",
    sub_observation: str = "",
    sub_action: str = "",
) -> dict:
    """
    Turn in place to face a target — measures the heading error itself via
    the external camera, so you don't need to estimate body_degrees by eye.

    If the target isn't visible, repositions once with navigate_to() (which
    scans and approaches automatically) and re-measures. If already within
    tolerance_deg, does nothing — no turn, no video/VQA call — and reports
    the measured angle. Otherwise issues a single turn() by exactly the
    measured angle.

    Falls back to the last target passed to navigate_to()/click_button()
    when target_class_yolo and target_class_free_text are both omitted.

    Like turn(), this is only for fine-grained corrections when already
    within camera range of the target: one uncorrected in-place turn, no
    obstacle avoidance, refused if the target is too far away (same
    near-target guard as turn()/drive()). For anything farther, call
    navigate_to() instead.

    Args:
        target_class_yolo:      YOLO class for the target (e.g. "cup"). Falls
                                back to the last target used by navigate_to/
                                click_button if omitted.
        target_class_free_text: Free-text description of the target (e.g.
                                "red plastic cup"). Same fallback as
                                target_class_yolo.
        tolerance_deg: Max heading error (degrees) treated as "already facing
                       the target" — below this, no turn is issued.
        speed:         Wheel speed, 15–20.
        expected:      Short description of the expected outcome
                       (auto-generated if blank).
        context:       Why this action is being taken and hints for evaluation.
        sub_observation: ~4-word video subtitle: what was just observed or instructed
                         (e.g. "Cup is to the left").
        sub_action:      ~4-word video subtitle: what the robot is doing now
                         (e.g. "Turning to face cup").
    """
    log.info(
        "[TOOL] turn_to yolo=%r free_text=%r tolerance_deg=%r",
        target_class_yolo, target_class_free_text, tolerance_deg,
    )
    yolo, free = _resolve_target(target_class_yolo, target_class_free_text)
    if not yolo and not free:
        return _err(
            "turn_to: no target specified and no prior target on record — "
            "pass target_class_yolo/target_class_free_text, or call "
            "navigate_to()/click_button() first."
        )
    if yolo and not free:
        return _err(
            "turn_to: target_class_free_text must be non-empty when target_class_yolo is set "
            "(otherwise the VLM fallback is silently skipped if YOLO finds nothing) — pass "
            "both explicitly, or pass neither to reuse the last navigate_to()/click_button() target."
        )
    if not (config.SPEED_MIN <= abs(speed) <= config.SPEED_MAX):
        return _err(f"speed must be between {config.SPEED_MIN} and {config.SPEED_MAX} (abs).")

    try:
        angle_deg, distance_mm = _measure_target(yolo, free)
        if angle_deg is None:
            log.info("turn_to: target not in view, repositioning")
            navigate_to(
                yolo, free,
                sub_observation="Target not in view",
                sub_action="Repositioning to target",
            )
            angle_deg, distance_mm = _measure_target(yolo, free)
    except Exception as exc:
        return _err(f"turn_to: alignment check failed: {exc}")

    if angle_deg is None:
        return _err(
            f"turn_to: could not locate target ({yolo or free}) to measure "
            "heading, even after navigate_to."
        )

    if abs(angle_deg) <= tolerance_deg:
        log.info("turn_to: already aligned within %.1f° (angle=%.1f°), no turn needed", tolerance_deg, angle_deg)
        return _ok({
            "message": f"already facing target within {tolerance_deg}° (measured {angle_deg:.1f}°); no turn needed",
            "measured_angle_deg": angle_deg,
            "measured_distance_mm": distance_mm,
        })

    log.info("turn_to: off by %.1f°, turning to face target", angle_deg)
    expected_str = expected if expected else f"robot rotates ~{abs(angle_deg):.0f}° to face the target"
    result = turn(
        angle_deg,
        speed=speed,
        expected=expected_str,
        context=context or "turn_to: turning to face measured target",
        sub_observation=sub_observation,
        sub_action=sub_action,
    )
    result["measured_angle_deg"] = angle_deg
    result["measured_distance_mm"] = distance_mm
    return result


@mcp.tool()
def drive_to(
    target_class_yolo: str = "",
    target_class_free_text: str = "",
    speed: int = 20,
    expected: str = "",
    context: str = "",
    sub_observation: str = "",
    sub_action: str = "",
) -> dict:
    """
    Drive straight toward a target by exactly its measured distance — you
    don't need to estimate duration_s by eye.

    Measures the straight-line distance to the target via the external
    camera (same px->mm body-plate calibration navigate_to()/click_button()
    use, then mm->wheel-encoder-degrees via navigation.mm_to_wheel_degrees).
    That measurement is robot-body-centroid to target-centroid, so driving
    the full distance would put the robot's centroid where the target's
    centroid currently is — the gripper, mounted forward of the centroid,
    would already have driven through the target well before that.
    drive_to() first subtracts config.DRIVE_TO_TOUCH_OFFSET_MM from the
    measured distance (floored at 0), on every leg — the short-range single
    drive, and both legs of the long-range auto-refine below — so the robot
    always targets the touch point, not the centroid. Unlike click_button()'s
    press distance, there is still no min/max clamp and no *added* overtravel
    margin.

    Long-range guard: a single blind drive accumulates dead-reckoning error
    (wheel slip/drift) in proportion to distance, so if the measured distance
    exceeds config.DRIVE_TO_LONG_RANGE_BODY_LENGTHS robot body lengths
    (default 2x ROBOT_BODY_LENGTH_MM), the first drive only covers
    config.DRIVE_TO_PARTIAL_FRACTION of the touch-adjusted distance (default
    85%) — the fraction is applied *after* the touch-offset subtraction, so
    this leg can't itself land inside the touch offset on a distance that's
    only just over the long-range threshold. drive_to() then re-measures from
    that closer, more reliable range and automatically fires one final drive
    to close the rest — capped at 2 physical drives total
    (each independently VQA-verified), so this never turns into an unbounded
    loop; that's what navigate_to() is for. The result carries
    `driven_distance_mm`, and — when a second drive ran — `drives_executed: 2`
    and `first_drive` (the first leg's own change_description/encoder
    readings), plus a `message` summarizing what happened. If the second
    measurement fails (target lost, or now out of near-target range), it
    falls back to reporting the first drive alone and asks the caller to
    re-invoke drive_to().

    The measured distance is the straight-line distance to the target as
    currently framed, so it only lands at the target if the robot is
    already facing it. If the target is off to one side, call turn_to()
    first.

    If the target isn't visible (or its distance can't be measured),
    repositions once with navigate_to() and re-measures. Falls back to the
    last target passed to navigate_to()/click_button() when
    target_class_yolo and target_class_free_text are both omitted.

    Like drive(), this is only for fine-grained approaches when already
    within camera range of the target: one uncorrected straight drive, no
    obstacle avoidance, refused if the target is too far away (same
    near-target guard as turn()/drive()). For anything farther, call
    navigate_to() instead.

    Args:
        target_class_yolo:      YOLO class for the target (e.g. "cup"). Falls
                                back to the last target used by navigate_to/
                                click_button if omitted.
        target_class_free_text: Free-text description of the target (e.g.
                                "red plastic cup"). Same fallback as
                                target_class_yolo.
        speed:    Wheel speed, 15–20. Positive = forward (toward the target).
        expected: Short description of the expected outcome (auto-generated
                  if blank).
        context:  Why this action is being taken and hints for evaluation.
        sub_observation: ~4-word video subtitle: what was just observed or instructed
                         (e.g. "Cup dead ahead").
        sub_action:      ~4-word video subtitle: what the robot is doing now
                         (e.g. "Driving to cup").
    """
    log.info(
        "[TOOL] drive_to yolo=%r free_text=%r speed=%r",
        target_class_yolo, target_class_free_text, speed,
    )
    yolo, free = _resolve_target(target_class_yolo, target_class_free_text)
    if not yolo and not free:
        return _err(
            "drive_to: no target specified and no prior target on record — "
            "pass target_class_yolo/target_class_free_text, or call "
            "navigate_to()/click_button() first."
        )
    if yolo and not free:
        return _err(
            "drive_to: target_class_free_text must be non-empty when target_class_yolo is set "
            "(otherwise the VLM fallback is silently skipped if YOLO finds nothing) — pass "
            "both explicitly, or pass neither to reuse the last navigate_to()/click_button() target."
        )
    if not (config.SPEED_MIN <= abs(speed) <= config.SPEED_MAX):
        return _err(f"speed must be between {config.SPEED_MIN} and {config.SPEED_MAX} (abs).")

    try:
        angle_deg, distance_mm = _measure_target(yolo, free)
        if distance_mm is None:
            log.info("drive_to: target/distance not measurable, repositioning")
            navigate_to(
                yolo, free,
                sub_observation="Target not in view",
                sub_action="Repositioning to target",
            )
            angle_deg, distance_mm = _measure_target(yolo, free)
    except Exception as exc:
        return _err(f"drive_to: distance check failed: {exc}")

    if distance_mm is None:
        return _err(
            f"drive_to: could not measure distance to target ({yolo or free}), "
            "even after navigate_to — the target may not be visible, or the "
            "robot's own body plate isn't visible for px->mm calibration."
        )

    guard = _target_too_far()
    if guard:
        return _err(guard)

    long_range_threshold_mm = config.DRIVE_TO_LONG_RANGE_BODY_LENGTHS * config.ROBOT_BODY_LENGTH_MM
    first_long_range = distance_mm > long_range_threshold_mm
    touch_adjusted_mm = max(0.0, distance_mm - config.DRIVE_TO_TOUCH_OFFSET_MM)
    if first_long_range:
        # Apply the dead-reckoning safety fraction to the touch-adjusted
        # distance, not the raw centroid distance — otherwise a distance only
        # just over the long-range threshold could have 85% of the *raw*
        # distance itself land inside the touch offset (e.g. 400mm raw: 0.85x
        # raw = 340mm driven, leaving only a 60mm centroid-gap — less than a
        # 130mm touch offset — i.e. already overshooting on this leg alone,
        # before the second leg's touch-aware logic even runs).
        first_drive_mm = touch_adjusted_mm * config.DRIVE_TO_PARTIAL_FRACTION
    else:
        first_drive_mm = touch_adjusted_mm

    first_drive_deg = int(round(nav_mod.mm_to_wheel_degrees(first_drive_mm)))
    if first_drive_deg <= 0:
        log.info(
            "drive_to: already within touch offset (measured distance=%.0fmm, "
            "touch offset=%.0fmm), no drive needed",
            distance_mm, config.DRIVE_TO_TOUCH_OFFSET_MM,
        )
        return _ok({
            "message": (
                f"already within touch offset (measured distance={distance_mm:.0f}mm, "
                f"touch offset={config.DRIVE_TO_TOUCH_OFFSET_MM:.0f}mm); no drive needed"
            ),
            "measured_angle_deg": angle_deg,
            "measured_distance_mm": distance_mm,
        })

    if first_long_range:
        log.info(
            "drive_to: measured distance=%.0fmm exceeds long-range threshold=%.0fmm "
            "(%.1fx body length) — driving %.0f%% of the %.0fmm touch-adjusted "
            "distance = %.0fmm first, then auto-refining with one final measured "
            "drive (drive_degrees=%d)",
            distance_mm, long_range_threshold_mm, config.DRIVE_TO_LONG_RANGE_BODY_LENGTHS,
            config.DRIVE_TO_PARTIAL_FRACTION * 100, touch_adjusted_mm, first_drive_mm, first_drive_deg,
        )
        first_desc = (
            f"drive_to speed={speed} drive_degrees={first_drive_deg} (driving "
            f"{first_drive_mm:.0f}mm of {distance_mm:.0f}mm measured "
            f"({touch_adjusted_mm:.0f}mm after touch offset); 1st of up to 2 "
            "drives to avoid overshoot)"
        )
        first_expected = expected if expected else (
            f"robot drives forward ~{first_drive_deg}° wheel rotation "
            f"(~{first_drive_mm:.0f}mm) — an intentional partial approach, stopping "
            f"well short of the full {distance_mm:.0f}mm measured distance to avoid "
            "overshoot; the robot will not yet be at the target"
        )
    else:
        log.info(
            "drive_to: measured distance=%.0fmm — driving %.0fmm after %.0fmm touch "
            "offset — drive_degrees=%d",
            distance_mm, first_drive_mm, config.DRIVE_TO_TOUCH_OFFSET_MM, first_drive_deg,
        )
        first_desc = (
            f"drive_to speed={speed} drive_degrees={first_drive_deg} (driving "
            f"{first_drive_mm:.0f}mm of {distance_mm:.0f}mm measured, stopping "
            f"{config.DRIVE_TO_TOUCH_OFFSET_MM:.0f}mm short to touch rather than "
            "center on the target)"
        )
        first_expected = expected if expected else (
            f"robot drives forward ~{first_drive_deg}° wheel rotation "
            f"(~{first_drive_mm:.0f}mm) toward the target, stopping with its front "
            "at the target rather than driving its center onto it"
        )

    first_result = _with_change_analysis(
        first_desc, first_expected,
        lambda: robot_mod.drive_degrees(first_drive_deg, speed, speed),
        context=context or "drive_to: driving measured distance to target",
        vqa_cameras={"simpleipcamera"},
        sub_observation=sub_observation,
        sub_action=sub_action,
    )
    first_result["measured_angle_deg"] = angle_deg
    first_result["measured_distance_mm"] = distance_mm

    if not first_long_range:
        return first_result

    if not first_result.get("ok"):
        # Drive failed (action_fn raised) — don't claim a driven distance we
        # don't actually know, and don't attempt a second drive on top of it.
        return first_result

    first_result["driven_distance_mm"] = first_drive_mm

    def _stop_after_first(reason: str) -> dict:
        first_result["partial_drive"] = True
        first_result["message"] = (
            f"First drive covered {first_drive_mm:.0f}mm of the {distance_mm:.0f}mm "
            f"measured distance (long-range guard: over "
            f"{config.DRIVE_TO_LONG_RANGE_BODY_LENGTHS:.0f}x body length = "
            f"{long_range_threshold_mm:.0f}mm). {reason}"
        )
        return first_result

    # ── Auto-refine: re-measure from the closer range and drive the rest ──────
    # Capped at one refinement (2 drives total) — auto-loop convenience, not
    # an unbounded closed loop (that's navigate_to's job). If the second
    # drive can't happen or can't fully close the gap, fall back to asking
    # the caller to invoke drive_to() again.
    try:
        _, second_distance_mm = _measure_target(yolo, free)
    except Exception as exc:
        return _stop_after_first(
            f"Could not re-measure for the second drive ({exc}) — call "
            "drive_to() again toward the same target to close the remaining distance."
        )

    if second_distance_mm is None:
        return _stop_after_first(
            "Target not visible for a second measurement — call drive_to() "
            "again toward the same target to close the remaining distance."
        )

    guard2 = _target_too_far()
    if guard2:
        return _stop_after_first(
            f"{guard2} The remaining approach needs navigate_to() instead of "
            "a second drive_to() drive."
        )

    second_drive_mm = max(0.0, second_distance_mm - config.DRIVE_TO_TOUCH_OFFSET_MM)
    second_drive_deg = int(round(nav_mod.mm_to_wheel_degrees(second_drive_mm)))
    if second_drive_deg <= 0:
        first_result["message"] = (
            f"First drive covered {first_drive_mm:.0f}mm; re-measured distance is "
            f"now {second_distance_mm:.0f}mm — within the "
            f"{config.DRIVE_TO_TOUCH_OFFSET_MM:.0f}mm touch offset, no second drive needed."
        )
        return first_result

    second_long_range = second_distance_mm > long_range_threshold_mm
    log.info(
        "drive_to: second (final) drive — re-measured distance=%.0fmm, driving "
        "%.0fmm after %.0fmm touch offset, drive_degrees=%d (drive 2 of 2, driven "
        "in full — no further auto-refine)",
        second_distance_mm, second_drive_mm, config.DRIVE_TO_TOUCH_OFFSET_MM, second_drive_deg,
    )
    second_desc = (
        f"drive_to speed={speed} drive_degrees={second_drive_deg} (2nd of 2 "
        f"drives, driving {second_drive_mm:.0f}mm of remeasured {second_distance_mm:.0f}mm, "
        f"stopping {config.DRIVE_TO_TOUCH_OFFSET_MM:.0f}mm short to touch rather "
        "than center on the target)"
    )
    second_expected = (
        f"robot drives forward ~{second_drive_deg}° wheel rotation "
        f"(~{second_drive_mm:.0f}mm) to reach the target, completing the "
        "approach the first drive intentionally left short, stopping with its "
        "front at the target rather than centered on it"
    )
    second_result = _with_change_analysis(
        second_desc, second_expected,
        lambda: robot_mod.drive_degrees(second_drive_deg, speed, speed),
        context=context or "drive_to: second drive closing remaining measured distance",
        vqa_cameras={"simpleipcamera"},
        sub_observation=sub_observation,
        sub_action=sub_action,
    )
    second_result["measured_angle_deg"] = angle_deg
    second_result["measured_distance_mm"] = distance_mm
    second_result["driven_distance_mm"] = first_drive_mm + second_drive_mm
    second_result["drives_executed"] = 2
    second_result["first_drive"] = {
        "driven_mm": first_drive_mm,
        "change_description": first_result.get("change_description"),
        "vqa_error": first_result.get("vqa_error"),
        "left": first_result.get("left"),
        "right": first_result.get("right"),
    }
    second_result["message"] = (
        f"Long-range drive ({distance_mm:.0f}mm measured, over "
        f"{config.DRIVE_TO_LONG_RANGE_BODY_LENGTHS:.0f}x body length) auto-refined "
        f"in 2 drives: {first_drive_mm:.0f}mm, then a re-measured "
        f"{second_distance_mm:.0f}mm (drove {second_drive_mm:.0f}mm after "
        f"{config.DRIVE_TO_TOUCH_OFFSET_MM:.0f}mm touch offset)."
        + (
            " The second drive was itself still long-range (2-drive cap reached) — "
            "verify the result and call drive_to() again if it undershot."
            if second_long_range else ""
        )
    )
    return second_result


def _square_up_to_target(
    target_class_yolo: str,
    target_class_free_text: str,
    tolerance_deg: float,
) -> tuple[dict | None, float | None]:
    """
    Square the robot's heading up to within *tolerance_deg* of the target so
    an upcoming forward press lands perpendicular to it.

    Measures the target's angle off the robot's forward heading via the
    external camera. If the target isn't visible at all, repositions with
    navigate_to() (which scans and approaches automatically) and re-measures.
    If visible but off-angle beyond tolerance_deg, issues a single turn()
    correction — CLAUDE.md reserves turn() for exactly this: a small heading
    fix once already within reach of the target.

    Returns (error_or_None, distance_mm). error_or_None is a dict on failure,
    else None once aligned (or already was). distance_mm is the most recent
    measured straight-line distance from the robot to the target — converted
    from pixels via navigation.mm_per_px(), the same body-plate calibration
    navigate_to() uses for its own drive distances — or None if it couldn't
    be computed (heading/target not detected, or body plate not visible).
    Re-measured after a turn correction, since the drive-forward distance
    should reflect the robot's post-turn position.
    """
    log.info("click_button: positioning — measuring alignment to switch")
    try:
        angle_deg, distance_mm = _measure_target(target_class_yolo, target_class_free_text)
        if angle_deg is None:
            log.info("click_button: positioning — switch not in view, repositioning")
            navigate_to(
                target_class_yolo, target_class_free_text,
                sub_observation="Switch not in view",
                sub_action="Repositioning to switch",
            )
            angle_deg, distance_mm = _measure_target(target_class_yolo, target_class_free_text)
    except Exception as exc:
        return _err(f"click_button: alignment check failed: {exc}"), None

    if angle_deg is None:
        return _err(
            f"click_button: could not locate the switch "
            f"({target_class_yolo or target_class_free_text}) to verify "
            "perpendicular alignment, even after navigate_to."
        ), None

    if abs(angle_deg) <= tolerance_deg:
        log.info("click_button: positioning — aligned within %.1f° (angle=%.1f°), no turn needed", tolerance_deg, angle_deg)
        return None, distance_mm

    log.info("click_button: positioning — off by %.1f°, squaring up with turn", angle_deg)
    result = turn(
        angle_deg,
        speed=config.SPEED_MIN,
        expected=f"robot rotates ~{abs(angle_deg):.0f}° to square up perpendicular to the switch",
        context="click_button pre-press alignment: squaring up to the switch before pressing",
        sub_observation="Switch off-angle",
        sub_action="Squaring to switch",
    )
    if not result.get("ok"):
        return result, distance_mm

    # Re-measure: turning in place can shift the camera's view of both the
    # robot's own body plate and the target enough to change the distance
    # estimate, so use the freshest reading for the upcoming press.
    _, distance_mm = _measure_target(target_class_yolo, target_class_free_text)
    return None, distance_mm


@mcp.tool()
def click_button(
    target_class_yolo: str,
    target_class_free_text: str,
    speed: int = 20,
    expected: str = "",
    context: str = "",
) -> dict:
    """
    Prepare and press a button/switch, then immediately release it.

    Before the press, the robot:
      1. Fully raises the arm and closes the gripper — clearing the switch
         and presenting a compact pressing profile. Not VQA-verified: this
         step is reliable enough that checking it would only cost time and
         a cloud API call (same reasoning as lift_arm's own VQA skip).
      2. Squares up to face the switch perpendicular: a single turn()
         correction if the switch is visible but off-angle, or navigate_to()
         first if it isn't visible at all. This also measures the straight-
         line distance to the switch.

    After the press (or after a failed alignment/press — see below), the
    robot reverses step 1: lowers the arm then opens the gripper, restoring
    grasp-ready pose. Also not VQA-verified, for the same reason as step 1.
    This matters beyond tidiness: a PDDL action modeled on this tool (e.g.
    toggle-lights) may not itself declare that pressing a button disturbs
    arm/gripper state, in which case a planner would believe the robot is
    still grasp-ready immediately after — restoring the pose here makes
    that belief true regardless of what the PDDL model does or doesn't
    know, instead of leaving the robot mid-press for whatever navigate/grasp
    runs next. Runs in a `finally`, so it still happens if alignment or the
    press itself fails — prep_for_press has already disturbed the pose by
    that point either way.

    The press distance is computed from that measured distance — converted to
    wheel-encoder degrees via the same px->mm body-plate calibration
    navigate_to() uses for its own drive distances (see navigation.mm_per_px /
    mm_to_wheel_degrees) — clamped to a sane range and given a small forward
    margin, rather than a fixed blind duration. The release drives back
    config.CLICK_RELEASE_FRACTION of that distance — just enough to clear
    the switch, without needing to return to the start position. See
    config.CLICK_PRESS_* for the margin/clamp/fallback constants.

    The press and release themselves still run inside a **single RPi Python
    script**, so there is no host round-trip and no VLM pause between them —
    guaranteeing the button is released immediately after the press,
    regardless of VLM validation latency.

    VQA verification of the press/release is currently **skipped by default**
    (config.CLICK_BUTTON_VQA=0): its verdicts on this action have proven
    unreliable, while the measured-distance press + fractional release is
    reliable enough for the current experiment. No `change_description` will
    be present in the result. Set CLICK_BUTTON_VQA=1 to re-enable.

    Args:
        target_class_yolo:      YOLO class for the button/switch (e.g. "button").
                                REQUIRED — must be non-empty.
        target_class_free_text: Free-text description (e.g. "white wall switch"),
                                used to verify heading and as a VLM fallback.
                                REQUIRED — must be non-empty.
        speed:               Wheel speed 15–20. Positive = forward (into button).
        expected:            What should visually happen during the press/release
                             (auto-generated if blank).
        context:             Why this action is being taken and evaluation hints.
    """
    log.info(
        "[TOOL] click_button yolo=%r free_text=%r speed=%r",
        target_class_yolo, target_class_free_text, speed,
    )
    if not target_class_yolo or not target_class_free_text:
        return _err(
            "target_class_yolo and target_class_free_text must both be non-empty — "
            "describe the button/switch so the robot can verify it is squarely "
            "facing it before pressing."
        )
    if not (config.SPEED_MIN <= abs(speed) <= config.SPEED_MAX):
        return _err(f"speed must be between {config.SPEED_MIN} and {config.SPEED_MAX} (abs).")

    # 1. Fully raise the arm + close the gripper — clears the switch and
    #    presents a compact pressing profile. No _with_change_analysis here:
    #    this step works reliably enough that the video capture + VQA call
    #    just cost time and a cloud API call for no real benefit — the
    #    actual press/release below still gets a full checked analysis.
    try:
        robot_mod.prep_for_press()
    except Exception as exc:
        return _err(f"click_button: prep_for_press failed: {exc}")

    try:
        # 2. Square up to the switch: navigate_to if not visible, else a single
        #    turn correction if off-angle beyond tolerance. Also yields the
        #    freshest measured distance to the switch.
        align_err, distance_mm = _square_up_to_target(
            target_class_yolo, target_class_free_text, config.CLICK_ALIGN_TOLERANCE_DEG,
        )
        if align_err is not None:
            return align_err

        # 3. Convert measured distance to a press distance: clamp + margin, same
        #    px->mm->wheel-degrees pipeline navigate_to uses (see docstring).
        if distance_mm is None:
            press_mm = config.CLICK_PRESS_FALLBACK_MM
            log.info("click_button: distance unmeasured — using fallback press distance %.0fmm", press_mm)
        else:
            clamped_mm = max(config.CLICK_PRESS_MIN_MM, min(distance_mm, config.CLICK_PRESS_MAX_MM))
            press_mm = clamped_mm + config.CLICK_PRESS_MARGIN_MM
            log.info(
                "click_button: measured distance=%.0fmm (clamped=%.0fmm) — press distance=%.0fmm (+%.0fmm margin)",
                distance_mm, clamped_mm, press_mm, config.CLICK_PRESS_MARGIN_MM,
            )
        press_degrees = int(round(nav_mod.mm_to_wheel_degrees(press_mm)))

        # 4. Press and release — single RPi script, one VQA call.
        desc = f"click_button speed={speed} press_degrees={press_degrees}"
        release_degrees = int(round(press_degrees * config.CLICK_RELEASE_FRACTION))
        expected_str = expected if expected else (
            f"robot drives forward ~{press_degrees}° wheel rotation (pressing button), "
            f"then immediately reverses ~{release_degrees}° (releasing, clearing the switch)"
        )
        return _with_change_analysis(
            desc, expected_str,
            lambda: robot_mod.click_button(speed, press_degrees),
            context=context,
            vqa_cameras={"simpleipcamera"},
            skip_vqa=not config.CLICK_BUTTON_VQA,
        )
    finally:
        # 5. Lower the arm + open the gripper — restores grasp-ready pose.
        #    Runs even on an early return above (alignment failure): prep_for_press
        #    already disturbed the pose, so it needs undoing either way. Not
        #    allowed to raise past this point — a cleanup failure shouldn't
        #    mask whatever the try block already decided to return.
        try:
            robot_mod.restore_grasp_pose()
        except Exception as exc:
            log.error("click_button: restore_grasp_pose failed: %s", exc, exc_info=True)


# ── arm ───────────────────────────────────────────────────────────────────────

@mcp.tool()
def move_arm(degrees: int, speed: int = config.DEFAULT_ARM_SPEED, expected: str = "", context: str = "",
             sub_observation: str = "", sub_action: str = "") -> dict:
    """
    Move the robot arm by the given number of degrees. Captures before/after
    images and returns a Gemini-generated `change_description`.

    Args:
        degrees:  How far to move. Positive = down, negative = up.
                  Start with values like ±30–90 and adjust based on results.
        speed:    Motor speed, 7-15 (default 7 — halved from the old default
                  of 15 to slow the move down for diagnosing the raise/lower-
                  then-fall bug; max 15 still caps jitter).
        expected: Short, precise description of the expected outcome
                  (e.g. "arm moves down ~45°, tip reaches ball height").
        context:  Why this action is being taken and hints for evaluation
                  (e.g. "positioning arm to grasp ball; last attempt stopped too high").
        sub_observation: ~4-word video subtitle: what was just observed or instructed
                         (e.g. "Arm too high").
        sub_action:      ~4-word video subtitle: what the robot is doing now
                         (e.g. "Lowering arm down").
    """
    log.info("[TOOL] move_arm degrees=%r speed=%r", degrees, speed)
    if not (config.ARM_SPEED_MIN <= abs(speed) <= config.ARM_SPEED_MAX):
        return _err(f"arm speed must be between {config.ARM_SPEED_MIN} and {config.ARM_SPEED_MAX} (abs).")
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
        annotate=False,
        sub_observation=sub_observation,
        sub_action=sub_action,
    )


@mcp.tool()
def lower_arm(speed: int = config.DEFAULT_ARM_SPEED, expected: str = "", context: str = "",
              sub_observation: str = "", sub_action: str = "") -> dict:
    """
    Lower the robot arm fully to ground level, then raise it 17° to keep the
    gripper clear of the floor and maximise wheel normal force. Captures
    before/after images and returns a Gemini-generated `change_description`.

    Args:
        speed:    Motor speed, 7-15 (default 7 — halved from the old default
                  of 15 to slow the move down for diagnosing the raise/lower-
                  then-fall bug; max 15 still caps jitter).
        expected: Short, precise description of the expected outcome.
        context:  Why this action is being taken and hints for evaluation.
        sub_observation: ~4-word video subtitle: what was just observed or instructed
                         (e.g. "Preparing to grab").
        sub_action:      ~4-word video subtitle: what the robot is doing now
                         (e.g. "Lowering arm down").
    """
    log.info("[TOOL] lower_arm speed=%r", speed)
    if not (config.ARM_SPEED_MIN <= abs(speed) <= config.ARM_SPEED_MAX):
        return _err(f"arm speed must be between {config.ARM_SPEED_MIN} and {config.ARM_SPEED_MAX} (abs).")
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
        skip_vqa=not config.LOWER_ARM_VQA,
        sub_observation=sub_observation,
        sub_action=sub_action,
    )


@mcp.tool()
def lift_arm(speed: int = config.LIFT_ARM_SPEED, expected: str = "", context: str = "",
             sub_observation: str = "", sub_action: str = "") -> dict:
    """
    Grasp-safe arm lift: closes the gripper with holding torque, raises the
    arm fully to the home/retracted position, holds briefly, then releases
    the gripper's hold torque. All four steps run inside a single script on
    the RPi (see mcp_robot.robot._GRASP_HOLD_AND_LIFT) so the hold torque
    stays actively applied by the BuildHAT firmware for the whole raise +
    settle window — this is what stops a grasped object (e.g. a cup) from
    slipping out while the arm moves. Captures before/after images and
    returns a Gemini-generated `change_description`.

    Args:
        speed:    Arm motor speed, 5-15 (default 5 — 66% of DEFAULT_ARM_SPEED,
                  slowed further so the raise-then-fall is easier to observe;
                  max 15 still caps jitter). Gripper close speed is fixed at
                  config.DEFAULT_GRIPPER_SPEED, not controlled by this arg.
        expected: Short, precise description of the expected outcome.
        context:  Why this action is being taken and hints for evaluation.
        sub_observation: ~4-word video subtitle: what was just observed or instructed
                         (e.g. "Arm still lowered").
        sub_action:      ~4-word video subtitle: what the robot is doing now
                         (e.g. "Raising arm up").
    """
    log.info("[TOOL] lift_arm speed=%r", speed)
    if not (config.ARM_SPEED_MIN <= abs(speed) <= config.ARM_SPEED_MAX):
        return _err(f"arm speed must be between {config.ARM_SPEED_MIN} and {config.ARM_SPEED_MAX} (abs).")
    expected_str = expected if expected else (
        "gripper jaws close fully (grasps anything between them), arm raises fully to "
        f"home position (~{config.ARM_UP_DEG}°, i.e. ~{config.ARM_DOWN_DEG - config.ARM_UP_DEG}° "
        "up from fully lowered) while the gripper holds, then the gripper releases its hold "
        "torque — fingers remain at the closed position, they do not reopen"
    )
    return _with_change_analysis(
        f"close gripper with hold, lift arm fully to home position at speed {speed}, "
        "hold, then release gripper hold",
        expected_str,
        lambda: robot_mod.lift_arm(speed),
        context=context,
        annotate=False,
        skip_vqa=not config.LIFT_ARM_VQA,
        sub_observation=sub_observation,
        sub_action=sub_action,
    )


# ── gripper ───────────────────────────────────────────────────────────────────

@mcp.tool()
def control_gripper(
    action: str,
    target_class_yolo: str,
    target_class_free_text: str,
    speed: int = 20,
    expected: str = "",
    context: str = "",
    sub_observation: str = "",
    sub_action: str = "",
) -> dict:
    """
    Open or close the gripper. Captures before/after images and returns a
    Gemini-generated `change_description`.

    Args:
        action:                 "open" or "close".
        target_class_yolo:      YOLO class for the grasp-readiness gate
                                (e.g. "cup", "ball", "any"). REQUIRED for
                                "close" — must be non-empty.
        target_class_free_text: Free-text description for Gemini Flash
                                (e.g. "red rubber ball"). REQUIRED for
                                "close" — must be non-empty.
        speed:                  Motor speed, 15–20.
        expected:               Short, precise description of the expected outcome
                                (e.g. "gripper closes around the ball").
        context:                Why this action is being taken and hints for
                                evaluation (e.g. "grasping paper ball; previous
                                close attempt slipped off").
        sub_observation: ~4-word video subtitle: what was just observed or instructed
                         (e.g. "Cup in position").
        sub_action:      ~4-word video subtitle: what the robot is doing now
                         (e.g. "Closing gripper").
    """
    log.info("[TOOL] control_gripper action=%r speed=%r yolo=%r free_text=%r",
             action, speed, target_class_yolo, target_class_free_text)
    if action == "close" and (not target_class_yolo or not target_class_free_text):
        return _err(
            "target_class_yolo and target_class_free_text must both be non-empty for action='close'. "
            "Describe the object to grasp (YOLO class + color/shape/material) so the "
            "grasp-readiness gate can verify it is in position."
        )
    if not (config.SPEED_MIN <= abs(speed) <= config.SPEED_MAX):
        return _err(f"speed must be between {config.SPEED_MIN} and {config.SPEED_MAX} (abs).")

    if action == "open":
        # Opening almost always succeeds — skip VQA to avoid the expense.
        try:
            return {"ok": True, "change_description": "Skipped description, because gripper 'open' almost always works", **robot_mod.control_gripper(action, speed)}
        except Exception as exc:
            return _err(str(exc))

    # Gate: check grasp readiness before closing.
    frame_result = cam_mod.capture_simpleipcamera_still(target_class_yolo=target_class_yolo, annotate=False)
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
        annotate=False,
        sub_observation=sub_observation,
        sub_action=sub_action,
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
    target_class_yolo: str,
    target_class_free_text: str,
) -> list[TextContent]:
    """
    CV-based grasp readiness check using the external (SimpleIPCamera) camera.

    Captures a live frame and verifies two conditions required before closing
    the gripper:
      1. The target object is touching the robot's front body.
      2. The green forward-arrow passes well over the object's center of mass.

    Args:
        target_class_yolo:      YOLO class to look for. Supported values:
                                  "cup"    — cup / bowl / bottle / vase
                                  "ball"   — sports ball / orange / apple
                                  "bottle" — bottle / cup / vase
                                  "any"    — most-forward object of any class
                                REQUIRED — must be non-empty.
        target_class_free_text: Free-text description sent to Gemini Flash when
                                YOLO finds nothing (e.g. "light switch").
                                REQUIRED — must be non-empty.

    Returns a verdict, a human-readable reason, and — when not ready — an
    actionable next step (drive closer, adjust heading, etc.).
    """
    log.info("[TOOL] check_grasp_readiness yolo=%r free_text=%r", target_class_yolo, target_class_free_text)
    if not target_class_yolo or not target_class_free_text:
        return [TextContent(type="text", text="ERROR: target_class_yolo and target_class_free_text must both be non-empty.")]
    try:
        frame_result = cam_mod.capture_simpleipcamera_still(target_class_yolo=target_class_yolo, annotate=False)
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
        log.error("[TOOL] check_grasp_readiness error: %s", exc, exc_info=True)
        return [TextContent(type="text", text=f"ERROR: {exc}")]


# ── navigation ────────────────────────────────────────────────────────────────

def _nav_track_motor(
    base_overlay: np.ndarray,
    stop_event: threading.Event,
) -> None:
    """Background thread for navigate_to: stream SimpleIPCamera frames with CV
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
        viz.log_nav_tracking(tracking, ts, reason="Tracking motion")

    cam_mod.stream_simpleipcamera_bgr(_on_frame, stop_event)


def _scan_for_target(
    target_class_yolo: str,
    target_class_free_text: str,
) -> tuple[bool, list[str], list[str]]:
    """Lower the arm and rotate up to SCAN_TOTAL_DEG, capturing front-camera
    frames at each step and running YOLO detection (no VLM — too expensive
    for a full sweep).

    Returns (found, frame_b64_list, log_lines).
    If found, the robot is left facing the direction where the target was seen.
    """
    step_deg = config.SCAN_STEP_DEG
    total_deg = config.SCAN_TOTAL_DEG
    n_steps = total_deg // step_deg

    logs: list[str] = ["--- Front-camera scan ---"]
    frames_b64: list[str] = []

    robot_mod.lower_arm()
    logs.append("Arm lowered for front-camera scan")

    rotated_so_far = 0
    for i in range(n_steps):
        if i > 0:
            robot_mod.turn(float(step_deg), config.SCAN_SPEED)
            rotated_so_far += step_deg
            logs.append(f"Scan step {i + 1}/{n_steps}: rotated +{step_deg}° CW "
                        f"(total {rotated_so_far}°)")

        try:
            still = cam_mod.capture_still()
            raw_bytes = base64.b64decode(still["frame"])
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            logs.append(f"Front camera capture failed at step {i + 1}: {exc}")
            continue

        if bgr is None:
            logs.append(f"Could not decode front camera frame at step {i + 1}")
            continue

        frames_b64.append(still["frame"])

        detected = None
        if target_class_yolo:
            try:
                objects = grasp_mod._yolo_detect(bgr, target_class=target_class_yolo)
                if objects:
                    detected = max(objects, key=lambda o: o.confidence)
            except Exception as exc:
                logs.append(f"YOLO error at scan step {i + 1}: {exc}")

        if detected is not None:
            logs.append(
                f"TARGET FOUND at scan step {i + 1} "
                f"(rotated {rotated_so_far}° CW): "
                f"'{detected.class_name}' conf={detected.confidence:.0%}"
            )
            log.info("[scan] target found after %d° CW rotation: %s conf=%.0f%%",
                     rotated_so_far, detected.class_name, detected.confidence * 100)
            return True, frames_b64, logs

        logs.append(f"Scan step {i + 1}/{n_steps}: target not visible")

    logs.append(f"Target not found after {total_deg}° scan")
    log.info("[scan] target not found after %d° sweep", total_deg)
    return False, frames_b64, logs


@mcp.tool()
def scan_for_target(
    target_class_yolo: str,
    target_class_free_text: str,
    sub_observation: str = "",
    sub_action: str = "",
) -> list[ImageContent | TextContent]:
    """
    Rotate up to 360° CW in 30° steps searching for the target using the
    front camera and YOLO. Call this when get_robot_state does not find
    the target — before navigate_to.

    The arm is lowered before sweeping so it does not block the camera.
    On success the robot is left facing the target, ready for navigate_to.

    Args:
        target_class_yolo:      YOLO class to detect (e.g. "cup", "ball").
        target_class_free_text: Free-text description used only for logs.
        sub_observation:        ~4-word subtitle: what was just observed.
        sub_action:             ~4-word subtitle: what the robot is doing.
    """
    log.info("[TOOL] scan_for_target yolo=%r free_text=%r sub_obs=%r sub_act=%r",
             target_class_yolo, target_class_free_text, sub_observation, sub_action)
    if not config.SCAN_ENABLED:
        log.info("[TOOL] scan_for_target mocked not-found (SCAN_ENABLED not set)")
        return [TextContent(type="text", text=(
            "Target not found after full sweep. "
            "Consider repositioning the robot or verifying the target class."
        ))]
    try:
        found, frames_b64, scan_logs = _scan_for_target(target_class_yolo, target_class_free_text)
        content: list[ImageContent | TextContent] = []
        content.append(TextContent(type="text", text="\n".join(scan_logs)))
        for b64 in frames_b64:
            content.append(_thumbnail_image_content(b64))
        if found:
            content.append(TextContent(type="text", text=(
                "Target found — robot is now facing it. "
                "Call navigate_to to approach."
            )))
        else:
            content.append(TextContent(type="text", text=(
                "Target not found after full sweep. "
                "Consider repositioning the robot or verifying the target class."
            )))
        return content
    except Exception as exc:
        log.error("[TOOL] scan_for_target error: %s", exc, exc_info=True)
        return [TextContent(type="text", text=f"ERROR: {exc}")]


def _capture_external_frame(target_class_yolo: str) -> tuple[np.ndarray | None, heading.Heading | None]:
    """Capture the current external (SimpleIPCamera) frame and detect robot heading.

    Shared by navigate_to's per-step preamble and its post-loop final-turn
    check — each applies its own handling when capture/decode/detection
    fails, so this only does the mechanical part and lets exceptions from
    the capture call itself propagate to the caller.

    Returns (bgr, h_result). bgr is None if the frame couldn't be decoded
    (h_result is then always None too); h_result is None if heading
    detection failed on an otherwise-valid frame.
    """
    frame_result = cam_mod.capture_simpleipcamera_still(target_class_yolo=target_class_yolo, annotate=False)
    raw_bytes = base64.b64decode(frame_result["frame"])
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    return bgr, heading.detect_heading(bgr)


def _execute_final_turn(
    face_deg: float | None,
    target_class_yolo: str,
    target_class_free_text: str,
    label: str,
) -> str | None:
    """Execute navigate_to's post-loop "face the target" turn.

    face_deg comes from a single obstacle-map estimate that can be several
    steps stale (only robot_px is re-tracked each step, not target_px), and
    large open-loop turns accumulate more encoder/floor-slip error than
    small ones — so a big turn can leave a meaningful residual heading
    error. When face_deg exceeds NAV_FINAL_TURN_REFINE_THRESHOLD_DEG,
    re-measure the heading with a fresh camera capture (the same
    _measure_target() helper turn_to() uses) and issue one small corrective
    turn if still off by more than TURN_TO_TOLERANCE_DEG.

    Returns a log-ready summary string, or None if face_deg is None
    (already facing the target — nothing to do).
    """
    if face_deg is None:
        return None
    direction = "CW" if face_deg > 0 else "CCW"
    log.info("[navigate_to] %s %+.0f° %s to face target", label, face_deg, direction)
    robot_mod.turn(float(face_deg), config.NAV_TURN_SPEED)
    msg = f"Final turn {face_deg:+.0f}° {direction} to face target"

    if abs(face_deg) <= config.NAV_FINAL_TURN_REFINE_THRESHOLD_DEG:
        return msg

    angle_deg, _ = _measure_target(target_class_yolo, target_class_free_text)
    if angle_deg is None or abs(angle_deg) <= config.TURN_TO_TOLERANCE_DEG:
        return msg

    refine_direction = "CW" if angle_deg > 0 else "CCW"
    log.info("[navigate_to] %s refinement turn %+.1f° %s (residual after large turn)",
             label, angle_deg, refine_direction)
    robot_mod.turn(float(angle_deg), config.NAV_TURN_SPEED)
    return msg + (f"; refinement turn {angle_deg:+.1f}° {refine_direction} "
                  "(residual heading error after large turn)")


@mcp.tool()
def navigate_to(
    target_class_yolo: str,
    target_class_free_text: str,
    max_steps: int = 10,
    sub_observation: str = "",
    sub_action: str = "",
) -> list[ImageContent | TextContent]:
    """
    Navigate the robot toward a target object using CV-based obstacle avoidance.

    Args (subtitle):
        sub_observation: ~4-word video subtitle: what was just observed or instructed
                         (e.g. "User said get cup").
        sub_action:      ~4-word video subtitle: what the robot is doing now
                         (e.g. "Navigating to cup").

    At every step the tool:
      1. Captures an external (SimpleIPCamera) frame and detects robot + target.
         If the target is not visible on the external camera, the robot
         lowers its arm and rotates up to 360° CW in 30° steps, capturing
         front-camera frames at each position and running YOLO + VLM
         detection. If the front camera spots the target, the robot
         re-checks the external camera from its new heading and continues
         navigation if the target is now visible there.
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
                                "bottle", "any"). REQUIRED — must be non-empty.
        target_class_free_text: Free-text description for Gemini Flash fallback
                                when YOLO finds nothing (e.g. "light switch").
                                REQUIRED — must be non-empty. Prefer a
                                description (color/shape/material) read from
                                the RAW camera frame — not from a debug/overlay
                                image, whose obstacle-mask tint can misrepresent
                                an object's color.
        max_steps:              Maximum navigation steps (default 6).
    """
    global _last_target_distance_px, _last_target_robot_radius_px
    log.info("[TOOL] navigate_to yolo=%r free_text=%r max_steps=%r",
             target_class_yolo, target_class_free_text, max_steps)
    if not target_class_yolo or not target_class_free_text:
        return [TextContent(type="text", text="ERROR: target_class_yolo and target_class_free_text must both be non-empty.")]
    _last_target_distance_px = None
    _last_target_robot_radius_px = None

    ts = time.strftime("%Y%m%d_%H%M%S")
    nav_dir = os.path.join(config.SNAPSHOT_DIR, f"navigate_to_{ts}") if config.SNAPSHOT_DIR else ""
    nav_t_start = time.time()

    key_frames_b64: list[str] = []
    step_logs: list[str] = []
    outcome = "max_steps_reached"
    debug_saved = False
    low_confidence_seen: str | None = None

    obs_map: nav_mod.ObstacleMap | None = None
    plan: nav_mod.NavPlan | None = None

    try:
        for step in range(max_steps):
            parts: list[str] = [f"=== Step {step + 1}/{max_steps} ==="]

            # ── 1. Capture external frame + heading ───────────────────────
            try:
                bgr, h_result = _capture_external_frame(target_class_yolo)
            except Exception as exc:
                parts.append(f"Camera capture failed: {exc}")
                step_logs.append("\n".join(parts))
                outcome = "camera_error"
                break

            if bgr is None:
                parts.append("Could not decode camera frame")
                step_logs.append("\n".join(parts))
                outcome = "camera_error"
                break

            # ── 2. Detect robot heading (cheap, every step) ───────────────
            if h_result is None:
                log.error("[navigate_to] step %d — robot heading not detected; aborting navigation", step + 1)
                parts.append("ERROR: robot heading not detected — navigation aborted")
                step_logs.append("\n".join(parts))
                outcome = "heading_not_detected"
                break
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
                    if isinstance(target_obj, vision.LowConfidenceDetection):
                        parts.append(str(target_obj))
                        low_confidence_seen = str(target_obj)
                        target_obj = None

                if target_obj is None:
                    if not config.SCAN_ENABLED:
                        log.info("[navigate_to] step %d — scan mocked not-found (SCAN_ENABLED not set)", step + 1)
                        parts.append(f"Target not detected on external camera "
                                     f"({target_class_yolo or target_class_free_text})")
                        step_logs.append("\n".join(parts))
                        step_logs.append("Target not found after full sweep. "
                                          "Consider repositioning the robot or verifying the target class.")
                        outcome = "target_not_detected"
                        break
                    parts.append(f"Target not detected on external camera "
                                 f"({target_class_yolo or target_class_free_text}) "
                                 f"— starting front-camera scan")
                    step_logs.append("\n".join(parts))

                    found, scan_frames, scan_logs = _scan_for_target(
                        target_class_yolo, target_class_free_text,
                    )
                    step_logs.extend(scan_logs)
                    key_frames_b64.extend(scan_frames)

                    if not found:
                        outcome = "target_not_detected"
                        break

                    # Re-capture external camera after scan rotation and
                    # re-detect the target — it may now be in frame from the
                    # new heading.
                    try:
                        frame_result = cam_mod.capture_simpleipcamera_still(target_class_yolo=target_class_yolo, annotate=False)
                        raw_bytes = base64.b64decode(frame_result["frame"])
                        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    except Exception:
                        pass

                    if bgr is not None:
                        h_result = heading.detect_heading(bgr)

                    if bgr is not None and target_class_yolo:
                        try:
                            objects = grasp_mod._yolo_detect(bgr, target_class=target_class_yolo)
                            if objects:
                                target_obj = (
                                    grasp_mod._pick_target(objects, h_result)
                                    if h_result is not None
                                    else max(objects, key=lambda o: o.confidence)
                                )
                        except Exception:
                            pass
                    if target_obj is None and bgr is not None and target_class_free_text:
                        target_obj = grasp_mod._vlm_detect(bgr, target_class_free_text)
                        if isinstance(target_obj, vision.LowConfidenceDetection):
                            step_logs.append(str(target_obj))
                            low_confidence_seen = str(target_obj)
                            target_obj = None

                    if target_obj is None:
                        step_logs.append(
                            "Front-camera scan found the target, but it is "
                            "still not visible on external camera after rotation "
                            "— cannot plan a path."
                        )
                        ok_enc, raw_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
                        if ok_enc:
                            key_frames_b64.append(base64.b64encode(raw_buf.tobytes()).decode())
                        outcome = "target_not_detected"
                        break

                    step_logs.append(
                        f"Post-scan: target '{target_obj.class_name}' "
                        f"now visible on external camera "
                        f"at {target_obj.center} conf={target_obj.confidence:.0%}"
                    )

                parts.append(f"Target '{target_obj.class_name}' "
                             f"at {target_obj.center} conf={target_obj.confidence:.0%}")
                if target_obj.note:
                    parts.append(f"VLM note: {target_obj.note}")

                # ── 4. Full obstacle detection — first step only ──────────
                obs_map = nav_mod.detect_obstacles(bgr, h_result, target_obj)
                free_frac = (obs_map.free_mask > 0).mean()
                parts.append(f"Obstacle map: {free_frac:.0%} free space")

                # ── 5. Path planning ──────────────────────────────────────
                plan = nav_mod.plan_path(obs_map, h_result)
                parts.append(f"Path: {plan.reason}")

                # ── 6. Save debug images ──────────────────────────────────
                if nav_dir:
                    try:
                        nav_mod.save_debug_images(bgr, obs_map, plan, nav_dir, step)
                        debug_saved = True
                    except Exception as exc:
                        log.warning("Failed to save nav debug images: %s", exc)

            else:
                # ── 3–5. Subsequent steps: track position, replan if needed ──
                robot_px = nav_mod.detect_robot_px(bgr)
                if robot_px is not None:
                    deviation = nav_mod.dist_to_path(robot_px, plan.path_px)
                    nav_mod.update_robot_position(obs_map, robot_px)
                    if deviation > obs_map.buffer_radius_px:
                        parts.append(
                            f"Deviation {deviation:.0f}px > buffer {obs_map.buffer_radius_px:.0f}px "
                            f"— replanning on existing C-space"
                        )
                        log.info(
                            "[navigate_to] step %d — replanning: deviation %.0fpx > buffer %.0fpx",
                            step + 1, deviation, obs_map.buffer_radius_px,
                        )
                        plan = nav_mod.plan_path(obs_map, h_result)
                        if nav_dir:
                            try:
                                nav_mod.save_debug_images(bgr, obs_map, plan, nav_dir, step,
                                                          suffix="_replan")
                                debug_saved = True
                            except Exception as exc:
                                log.warning("Failed to save replan debug images: %s", exc)
                    else:
                        nav_mod.refresh_approach_path(obs_map, plan)
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

            # Both overlay and cspace are external-camera renders (navigation
            # annotations are always external-camera based), so they're
            # combined side-by-side into one frame for the single "Annotated
            # — External Camera" row rather than split across rows.
            cspace_bgr = nav_mod.build_cspace_bgr(bgr, obs_map, plan)
            nav_panel_bgr = cv2.hconcat([overlay_bgr, cspace_bgr])
            ok_nav, nav_buf = cv2.imencode(".jpg", nav_panel_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok_nav:
                viz.log_annotated_images(
                    external_b64=base64.b64encode(nav_buf.tobytes()).decode(),
                    reason="Navigation plan",
                )

            # ── 8. Termination checks ─────────────────────────────────────
            if nav_mod.near_target(obs_map):
                face_deg = nav_mod.turn_to_face_target(obs_map, h_result)
                turn_msg = _execute_final_turn(
                    face_deg, target_class_yolo, target_class_free_text, "final turn"
                )
                if turn_msg is not None:
                    parts.append(turn_msg)
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
            turn_deg, drive_deg, reverse = nav_mod.commands_for_step(obs_map, plan, h_result)
            direction_label = "reverse" if reverse else "forward"
            parts.append(f"Commands: turn={turn_deg:+.0f}°, drive={drive_deg:.0f}° (wheel, {direction_label})")

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
                if reverse:
                    log.info("[navigate_to] step %d — reversing %.0f° (encoder) "
                             "— obstacle ahead", step + 1, drive_deg)
                    robot_mod.drive_degrees(int(round(drive_deg)), -config.SPEED_MAX, -config.SPEED_MAX)
                else:
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

            # ── Bonus final step: face the target even though we didn't ──
            # arrive, so the caller isn't left staring at a random heading.
            bgr, h_result = _capture_external_frame(target_class_yolo)
            robot_px = nav_mod.detect_robot_px(bgr) if bgr is not None else None
            if robot_px is not None:
                nav_mod.update_robot_position(obs_map, robot_px)
            face_deg = (
                nav_mod.turn_to_face_target(obs_map, h_result)
                if h_result is not None else None
            )
            turn_msg = _execute_final_turn(
                face_deg, target_class_yolo, target_class_free_text,
                "max_steps reached — final turn",
            )
            if turn_msg is not None:
                step_logs.append(f"{turn_msg} (max_steps reached)")

    except Exception as exc:
        log.error("[TOOL] navigate_to error: %s", exc, exc_info=True)
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
        "success":              "Navigation successful — robot is at the target.",
        "path_blocked":         "Navigation failed — no obstacle-free path found.",
        "target_not_detected":  (
            "Navigation aborted — a candidate target was seen but below the "
            "confidence threshold required to act (see log for certainty achieved)."
            if low_confidence_seen else
            "Navigation aborted — target not detected (YOLO/VLM found nothing)."
        ),
        "heading_not_detected": "Navigation aborted — robot heading could not be detected (yellow body not visible).",
        "camera_error":         "Navigation aborted — camera error.",
        "error":                "Navigation aborted — unexpected error.",
        "max_steps_reached":    f"Navigation incomplete — max_steps ({max_steps}) reached without reaching target.",
    }.get(outcome, outcome)

    if outcome in ("success", "max_steps_reached"):
        outcome_text += (
            " If you check the frame(s) above and confirm there are no further "
            "obstacles between the robot and the target, consider drive_to()/"
            "turn_to() for the remaining approach instead of calling navigate_to() "
            "again — they measure the exact angle/distance to the target directly "
            "from the camera and skip the full replanning loop."
        )

    log_text = "\n\n".join(step_logs) if step_logs else "(no steps executed)"
    if nav_dir and debug_saved:
        log_text += f"\n\nDebug images: {nav_dir}"
    content.append(
        TextContent(type="text", text=f"Navigate-to outcome: {outcome_text}\n\n{log_text}")
    )

    if sub_observation or sub_action:
        from mcp_robot.recorder import get_recorder
        rec = get_recorder()
        nav_meta = {"tool": f"navigate_to {target_class_yolo}",
                    "sub_observation": sub_observation or None,
                    "sub_action": sub_action or None}
        for cam in ("simpleipcamera", "pi_camera"):
            rec.tag_range(cam, nav_t_start, time.time(), nav_meta)

    return content


# ── VLM object localization ───────────────────────────────────────────────────

@mcp.tool()
def locate_object(description: str) -> list[ImageContent | TextContent]:
    """
    Locate an arbitrary object using Gemini Flash vision localization.

    Use this for objects that YOLO cannot detect — anything outside the COCO-80
    class set, such as "light switch", "door handle", "power outlet",
    "red cable connector", "white box on the shelf", etc.

    Captures the external (SimpleIPCamera) camera, asks Gemini Flash to find the
    object described by *description*, and returns:
      • An annotated frame with the heading arrow, object bounding box, and
        angle label overlaid.
      • The angle from the robot's current forward direction to the object
        (positive = CW, negative = CCW, viewed from above).
      • A short note from the VLM describing what it found.

    Pass the returned angle directly to `turn(body_degrees=<angle>)` to face
    the object before driving toward it.

    If a candidate is seen but below the confidence threshold required to act,
    this reports the achieved certainty (e.g. "Only 82% certainty...") instead
    of claiming the object was not found — reposition for a clearer view and
    retry rather than treating it as a definite absence.

    Args:
        description: Free-text description of the object to find, e.g.
                     "light switch", "door handle", "yellow power strip".
    """
    log.info("[TOOL] locate_object description=%r", description)
    try:
        # Capture external camera (unannotated — VLM should see the raw scene)
        frame_result = cam_mod.capture_simpleipcamera_still(target_class_yolo="", annotate=False)
        raw = base64.b64decode(frame_result["frame"])
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return [TextContent(type="text", text="ERROR: could not decode external camera frame.")]

        # Locate the object with the VLM→CV hybrid pipeline
        low_confidence: vision.LowConfidenceDetection | None = None
        parse_error: vision.VQAResponseParseError | None = None
        try:
            vlm_result = vision.locate_object_hybrid(bgr, description)
        except vision.VQAResponseParseError as exc:
            vlm_result = None
            parse_error = exc
        if isinstance(vlm_result, vision.LowConfidenceDetection):
            low_confidence = vlm_result
            vlm_result = None

        if vlm_result is None:
            annotated_bgr = heading.annotate_bgr(bgr)
            ok, buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buf.tobytes()).decode() if ok else frame_result["frame"]
            if parse_error is not None:
                text = f"VQA failure while locating '{description}': {parse_error}"
            elif low_confidence is not None:
                text = str(low_confidence)
            else:
                text = (
                    f"Object not found: '{description}' was not detected in the external "
                    "camera frame. Check that the object is visible and try again."
                )
            return [
                _image_content(frame_b64),
                TextContent(type="text", text=text),
            ]

        (x1, y1, x2, y2), obj_center, confidence, note, _rough_bbox = vlm_result

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
        log.error("[TOOL] locate_object error: %s", exc, exc_info=True)
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
        viz.log_annotated_images(pi_b64=result["frame"], reason="Manual capture")
        path_info = f" — saved to {result['path']}" if result.get("path") else ""
        return [
            _image_content(result["frame"]),
            TextContent(
                type="text",
                text=f"Pi Camera — {result['width']}×{result['height']} JPEG ({result['bytes']} bytes){path_info}",
            ),
        ]
    except Exception as exc:
        log.error("[TOOL] get_front_camera_image error: %s", exc, exc_info=True)
        return [TextContent(type="text", text=f"ERROR: {exc}")]


@mcp.tool()
def get_external_camera_image() -> list[ImageContent | TextContent]:
    """
    Capture a single still frame from the SimpleIPCamera (third-person/overhead view).
    Useful for observing the robot's position and surroundings from outside.
    """
    log.info("[TOOL] get_external_camera_image")
    try:
        result = cam_mod.capture_simpleipcamera_still(target_class_yolo="")
        viz.log_annotated_images(external_b64=result["frame"], reason="Manual capture")
        path_info = f" — saved to {result['path']}" if result.get("path") else ""
        return [
            _image_content(result["frame"]),
            TextContent(type="text", text=f"SimpleIPCamera — external/third-person view{path_info}"),
        ]
    except Exception as exc:
        log.error("[TOOL] get_external_camera_image error: %s", exc, exc_info=True)
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
        log.error("[TOOL] capture_front_video_clip error: %s", exc, exc_info=True)
        return [TextContent(type="text", text=f"ERROR: {exc}")]


@mcp.tool()
def capture_external_video_clip(
    duration_s: float = 2.0,
    fps: float = 2.0,
) -> list[ImageContent | TextContent]:
    """
    Capture a short clip from the SimpleIPCamera (third-person/overhead view).

    Args:
        duration_s: Clip length in seconds (1–10 recommended).
        fps:        Frames per second (1–5 recommended).
    """
    log.info("[TOOL] capture_external_video_clip duration_s=%r fps=%r", duration_s, fps)
    try:
        result = cam_mod.capture_simpleipcamera_clip(duration_s, fps)
        content: list[ImageContent | TextContent] = [
            TextContent(
                type="text",
                text=f"SimpleIPCamera — {result['count']} frames at {fps:.1f} fps ({duration_s}s)",
            )
        ]
        for frame_b64 in result["frames"]:
            content.append(_clip_image_content(frame_b64))
        vqa = vision.describe_clip(
            "simpleipcamera", result["frames"], result.get("paths"),
            raw_frames=result.get("raw_frames"),
        )
        if vqa:
            content.append(TextContent(type="text", text=f"Clip VQA (Qwen):\n{vqa}"))
        return content
    except Exception as exc:
        log.error("[TOOL] capture_external_video_clip error: %s", exc, exc_info=True)
        return [TextContent(type="text", text=f"ERROR: {exc}")]


@mcp.tool()
def get_robot_state(
    target_class_yolo: str,
    target_class_free_text: str,
) -> list[ImageContent | TextContent]:
    """
    One-shot state snapshot: all motor positions + live frames from both
    cameras (SimpleIPCamera = wider third-person view; Pi Camera = front/robot-eye view).
    Call this before planning any sequence of actions.

    Args:
        target_class_yolo:      YOLO class to detect and annotate in the external
                                camera frame. Supported values:
                                  "cup"    → cup, bowl, bottle, vase
                                  "ball"   → sports ball, orange, apple
                                  "bottle" → bottle, cup, vase
                                  "any"    → most-forward object of any class
                                No default — state explicitly what you're
                                looking for, or pass "" to skip YOLO.
        target_class_free_text: Free-text description for Gemini Flash when YOLO
                                finds nothing (e.g. "light switch", "door handle").
                                The detected object's angle from the robot's heading
                                is returned so you know how much to turn. No default —
                                REQUIRED whenever target_class_yolo is non-empty (the
                                VLM fallback is silently skipped otherwise); pass ""
                                for both to skip object search and just check state.
    """
    global _state_call_count, _last_target_distance_px, _last_target_robot_radius_px
    global _last_target_yolo, _last_target_free_text
    log.info("[TOOL] get_robot_state yolo=%r free_text=%r", target_class_yolo, target_class_free_text)
    if target_class_yolo and not target_class_free_text:
        return [TextContent(type="text", text=(
            "ERROR: target_class_free_text must be non-empty when target_class_yolo is set — "
            "otherwise the VLM fallback is silently skipped if YOLO finds nothing. "
            "Pass both, or pass target_class_yolo=\"\" too to skip object search entirely."
        ))]
    try:
        _state_call_count += 1
        content: list[ImageContent | TextContent] = []
        try:
            simpleipcam_frame = cam_mod.capture_simpleipcamera_still(
                target_class_yolo=target_class_yolo,
                target_class_free_text=target_class_free_text,
            )
            viz.log_annotated_images(external_b64=simpleipcam_frame["frame"], reason="State check")
            content.append(TextContent(type="text", text="Third-person view 320×240 thumbnail:"))
            content.append(_thumbnail_image_content(simpleipcam_frame["frame"]))
            angle_deg = simpleipcam_frame.get("object_angle_deg")
            vlm_note = simpleipcam_frame.get("vlm_note")
            _last_target_distance_px = simpleipcam_frame.get("object_distance_px")
            _last_target_robot_radius_px = simpleipcam_frame.get("robot_radius_px")
            _last_target_yolo = target_class_yolo
            _last_target_free_text = target_class_free_text
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
                if target_class_yolo or target_class_free_text:
                    content.append(TextContent(type="text", text=(
                        f"Target not found in current view "
                        f"({target_class_yolo or target_class_free_text}). "
                        f"Call scan_for_target(target_class_yolo={target_class_yolo!r}, "
                        f"target_class_free_text={target_class_free_text!r}) "
                        f"to rotate and search before navigate_to."
                    )))
            if vlm_note:
                content.append(TextContent(type="text", text=f"VLM note: {vlm_note}"))
        except Exception as exc:
            log.error("[TOOL] get_robot_state simpleipcamera capture error: %s", exc, exc_info=True)
            _last_target_distance_px = None
            _last_target_robot_radius_px = None
            content.append(TextContent(type="text", text=f"Third-person view unavailable: {exc}"))
        pi_frame = cam_mod.capture_still()
        viz.log_annotated_images(pi_b64=pi_frame["frame"], reason="State check")
        content.append(TextContent(type="text", text="Front view (Pi Camera):"))
        content.append(_thumbnail_image_content(pi_frame["frame"]))
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
        log.error("[TOOL] get_robot_state error: %s", exc, exc_info=True)
        return [TextContent(type="text", text=f"ERROR: {exc}")]


# ── task video ────────────────────────────────────────────────────────────────

@mcp.tool()
def compile_video(since: str, camera: str = "simpleipcamera") -> dict:
    """
    Compile a video by concatenating recorded motion segments since a given
    timestamp. Segments are produced continuously by the SegmentRecorder and
    contain only motion-bounded footage (no stale/static frames).

    Call this after a sequence of actions to get a single video of the task.

    Args:
        since:  UNIX timestamp as a string (e.g. "1746613200.0"), or a legacy
                "YYYY-MM-DD HH:MM:SS[,ms]" / "YYYYMMDD_HHMMSS" string.
        camera: "simpleipcamera" (default) or "pi_camera" for a single-camera clip,
                or "merged" to tile both cameras plus a subtitle card
                (observation, then action) into one video: subtitle
                top-left, Pi camera bottom-left, SimpleIPCamera full-height right.

    Returns a dict with video_path, segment_count, total_duration_s.
    """
    log.info("[TOOL] compile_video since=%r camera=%r", since, camera)
    from mcp_robot.video_compiler import compile_merged_video, compile_task_video
    if camera == "merged":
        result = compile_merged_video(since, config.SEGMENT_MANIFEST)
    else:
        result = compile_task_video(since, config.SEGMENT_MANIFEST, camera)
    if not result.ok:
        return _err(result.error)
    if result.video_path is None:
        return _ok({"message": "No segments found since given timestamp", "segment_count": 0})
    return _ok({"video_path": result.video_path, "segment_count": result.segment_count,
                "total_duration_s": result.total_duration_s})


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


def _run_simpleipcamera() -> None:
    reported = [False]

    def _on_frame(frame: str, ts: float) -> None:
        if not reported[0]:
            reported[0] = True
            _log_init_progress("simpleipcamera", "done")
        viz.log_simpleipcamera_frame(frame, ts)

    backoff = 1.0
    while not _stop.is_set():
        try:
            cam_mod.stream_simpleipcamera(stop_event=_stop, on_frame=_on_frame)
        except Exception as exc:
            if not reported[0]:
                _log_init_progress("simpleipcamera", "failed")
            log.warning("SimpleIPCamera stream ended: %s", exc)
        if not _stop.is_set():
            log.info("SimpleIPCamera reconnecting in %.0fs...", backoff)
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
        (_run_simpleipcamera,   "simpleipcamera"),
    ]:
        threading.Thread(target=target, name=name, daemon=True).start()

    atexit.register(_shutdown)


def _shutdown() -> None:
    _stop.set()
    viz.flush()
    from mcp_robot.recorder import get_recorder
    get_recorder().close()


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
