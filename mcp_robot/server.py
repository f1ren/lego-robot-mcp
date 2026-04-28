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
  drive                  forward / backward / left / right / stop

  Arm & gripper
  ─────────────
  move_arm               Move arm up or down
  control_gripper        Open or close the gripper

  High-level actions
  ──────────────────
  grasp                  Lower arm + close gripper
  put                    Open gripper + raise arm

  Camera
  ──────
  capture_image          Capture one still frame (returns ImageContent)
  capture_video_clip     Capture N-second clip, return frames as images

Run with:
    python3 -m mcp_robot.server
"""
from __future__ import annotations

import atexit
import logging
import threading

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

import mcp_robot.camera as cam_mod
import mcp_robot.robot  as robot_mod
from mcp_robot import config, viz, vision

log = logging.getLogger(__name__)
_stop = threading.Event()

mcp = FastMCP(
    "lego-robot",
    instructions=(
        "Control a 4-motor Lego robot via BuildHat on a Raspberry Pi. "
        "Motors: left_wheel (A), right_wheel (B), gripper (C), arm (D). "
        "Always call get_robot_state before planning a sequence of actions. "
        "Motor-action tools (move_motor, drive, move_arm, control_gripper, "
        "grasp, put) automatically capture before/after images and return a "
        "Gemini-generated `change_description` summarising what changed — "
        "you do NOT need to call capture_image afterwards to verify them. "
        "Use capture_image / capture_video_clip / get_robot_state when you "
        "explicitly need to see the scene. "
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


def _with_change_analysis(action_desc: str, expected: str, action_fn) -> dict:
    """
    Capture before-frames, run *action_fn*, capture after-frames, and return
    the action result merged with a Gemini-generated `change_description`
    that includes a verdict on whether *expected* was achieved.

    On action error, returns _err(...) and skips the after-capture / Gemini call.
    On vision failure, the action result is still returned (description omitted).
    """
    before, before_paths = _capture_pair("before")
    try:
        result = action_fn()
    except Exception as exc:
        return _err(str(exc))
    after, after_paths = _capture_pair("after")
    description = vision.describe_change(
        action_desc, expected, before, after, before_paths, after_paths
    )
    out = _ok(result)
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
    direction: str,
    duration_s: float = 1.0,
    speed: int = 50,
) -> dict:
    """
    Drive the robot. Captures before/after images and returns a Gemini-generated
    `change_description` alongside motor positions.

    Args:
        direction: "forward" | "backward" | "left" | "right" | "stop"
        duration_s: How long to run the wheels (seconds). Ignored for "stop".
        speed:      Wheel speed, 1–100.
    """
    desc = (
        f"drive {direction}"
        if direction == "stop"
        else f"drive {direction} for {duration_s}s at speed {speed}"
    )
    expected = {
        "forward":  "robot translates forward — droidcam shows the body moving forward; pi_camera front view shifts to show new content ahead; gripper/arm unchanged",
        "backward": "robot translates backward — droidcam shows the body moving backward; pi_camera front view shifts to show what was previously behind; gripper/arm unchanged",
        "left":     "robot pivots left in place — droidcam shows orientation rotating counter-clockwise; pi_camera view pans accordingly",
        "right":    "robot pivots right in place — droidcam shows orientation rotating clockwise; pi_camera view pans accordingly",
        "stop":     "robot stops — no visible position or orientation change between frames",
    }.get(direction, f"robot performs {direction!r} motion")
    return _with_change_analysis(
        desc, expected, lambda: robot_mod.drive(direction, duration_s, speed),
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
def grasp() -> dict:
    """
    High-level GRASP: lower arm then close gripper. Captures before/after
    images and returns a Gemini-generated `change_description` confirming
    whether the grip succeeded.
    """
    return _with_change_analysis(
        "grasp (lower arm + close gripper)",
        "arm lowers toward an object directly in front of the robot, then gripper "
        "jaws close around it — in the AFTER frames the gripper should be closed "
        "and (if an object was present) the object should appear held between the jaws",
        robot_mod.grasp,
    )


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
def capture_image() -> list[ImageContent | TextContent]:
    """
    Capture a single still frame from the Pi Camera.
    Returns the image so you can inspect the scene visually.
    """
    try:
        result = cam_mod.capture_still()
        return [
            _image_content(result["frame"]),
            TextContent(
                type="text",
                text=f"Captured {result['width']}×{result['height']} JPEG ({result['bytes']} bytes)",
            ),
        ]
    except Exception as exc:
        return [TextContent(type="text", text=f"ERROR: {exc}")]


@mcp.tool()
def capture_video_clip(
    duration_s: float = 2.0,
    fps: float = 2.0,
) -> list[ImageContent | TextContent]:
    """
    Capture a short video clip as a sequence of JPEG frames.

    Args:
        duration_s: Clip length in seconds (1–10 recommended).
        fps:        Frames per second (1–5 recommended for SSH bandwidth).
    """
    try:
        result = cam_mod.capture_clip(duration_s, fps)
        content: list[ImageContent | TextContent] = [
            TextContent(
                type="text",
                text=f"Captured {result['count']} frames at {fps:.1f} fps — {duration_s}s clip",
            )
        ]
        for frame_b64 in result["frames"]:
            content.append(_image_content(frame_b64))
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
    backoff = 1.0
    while not _stop.is_set():
        try:
            cam_mod.stream_live(fps=5.0, stop_event=_stop)
        except Exception as exc:
            log.warning("Pi Camera stream ended: %s", exc)
        if not _stop.is_set():
            log.info("Pi Camera reconnecting in %.0fs...", backoff)
            _stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)
        else:
            break


def _run_droidcam() -> None:
    backoff = 1.0
    while not _stop.is_set():
        try:
            cam_mod.stream_droidcam(stop_event=_stop)
        except Exception as exc:
            log.warning("DroidCam stream ended: %s", exc)
        if not _stop.is_set():
            log.info("DroidCam reconnecting in %.0fs...", backoff)
            _stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)
        else:
            break


def _start_background_streams() -> None:
    log.info("Initializing motors...")
    try:
        robot_mod.get_all_positions()
        log.info("Motors ready.")
    except Exception as exc:
        log.warning("Motor init failed (%s) — continuing without motor data.", exc)

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
