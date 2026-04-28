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
from mcp_robot import config, viz

log = logging.getLogger(__name__)
_stop = threading.Event()

mcp = FastMCP(
    "lego-robot",
    instructions=(
        "Control a 4-motor Lego robot via BuildHat on a Raspberry Pi. "
        "Motors: left_wheel (A), right_wheel (B), gripper (C), arm (D). "
        "Always call get_robot_state before planning a sequence of actions. "
        "After each action call capture_image or capture_video_clip to confirm the outcome visually. "
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
    try:
        return _ok(robot_mod.move_motor(port.upper(), degrees, speed))
    except Exception as exc:
        return _err(str(exc))


# ── wheel driving ─────────────────────────────────────────────────────────────

@mcp.tool()
def drive(
    direction: str,
    duration_s: float = 1.0,
    speed: int = 50,
) -> dict:
    """
    Drive the robot.

    Args:
        direction: "forward" | "backward" | "left" | "right" | "stop"
        duration_s: How long to run the wheels (seconds). Ignored for "stop".
        speed:      Wheel speed, 1–100.
    """
    try:
        return _ok(robot_mod.drive(direction, duration_s, speed))
    except Exception as exc:
        return _err(str(exc))


# ── arm ───────────────────────────────────────────────────────────────────────

@mcp.tool()
def move_arm(degrees: int, speed: int = 30) -> dict:
    """
    Move the robot arm by the given number of degrees.

    Args:
        degrees: How far to move. Positive = down, negative = up.
                 Start with values like ±30–90 and adjust based on results.
        speed:   Motor speed, 1–100.
    """
    try:
        return _ok(robot_mod.move_arm(degrees, speed))
    except Exception as exc:
        return _err(str(exc))


# ── gripper ───────────────────────────────────────────────────────────────────

@mcp.tool()
def control_gripper(action: str, speed: int = 25) -> dict:
    """
    Open or close the gripper.

    Args:
        action: "open" or "close".
        speed:  Motor speed, 1–100.
    """
    try:
        return _ok(robot_mod.control_gripper(action, speed))
    except Exception as exc:
        return _err(str(exc))


# ── compound actions ──────────────────────────────────────────────────────────

@mcp.tool()
def grasp() -> dict:
    """
    High-level GRASP: lower arm then close gripper.
    Call verify_action or analyze_scene afterwards to confirm grip.
    """
    try:
        return _ok(robot_mod.grasp())
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def put() -> dict:
    """
    High-level PUT: open gripper then raise arm.
    Call verify_action or analyze_scene afterwards to confirm release.
    """
    try:
        return _ok(robot_mod.put())
    except Exception as exc:
        return _err(str(exc))


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
