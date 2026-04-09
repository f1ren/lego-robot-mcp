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

  Camera & vision
  ────────────────
  capture_image          Capture one still frame (returns ImageContent)
  capture_video_clip     Capture N-second clip, return frames as images
  analyze_scene          Capture frame + ask Gemini a question
  analyze_video_clip     Capture clip + ask Gemini (temporal reasoning)
  verify_action          Before/after frames → Gemini success judgement

Run with:
    python3 -m mcp_robot.server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

import mcp_robot.camera as cam_mod
import mcp_robot.robot  as robot_mod
import mcp_robot.vision as vision_mod

mcp = FastMCP(
    "lego-robot",
    instructions=(
        "Control a 4-motor Lego robot via BuildHat on a Raspberry Pi. "
        "Motors: left_wheel (A), right_wheel (B), arm (C), gripper (D). "
        "Always call get_robot_state before planning a sequence of actions. "
        "After each action call verify_action or analyze_scene to confirm outcome. "
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
def move_arm(
    direction: str,
    degrees: int | None = None,
    speed: int = 30,
) -> dict:
    """
    Move the robot arm.

    Args:
        direction: "up" or "down".
        degrees:   How far to move (omit for full configured range).
        speed:     Motor speed, 1–100.
    """
    try:
        return _ok(robot_mod.move_arm(direction, degrees, speed))
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


# ── camera + Gemini ───────────────────────────────────────────────────────────

@mcp.tool()
def analyze_scene(prompt: str = "Describe what you see.") -> list[ImageContent | TextContent]:
    """
    Capture a still frame and ask Gemini to analyse it.

    Args:
        prompt: Question or instruction for Gemini, e.g.
                "Is the gripper holding an object?",
                "Where is the red block relative to the robot?"
    """
    try:
        result = cam_mod.capture_still()
        analysis = vision_mod.analyze_frame(result["frame"], prompt)
        return [
            _image_content(result["frame"]),
            TextContent(type="text", text=analysis),
        ]
    except Exception as exc:
        return [TextContent(type="text", text=f"ERROR: {exc}")]


@mcp.tool()
def analyze_video_clip(
    prompt: str = "Describe what happens across these frames.",
    duration_s: float = 2.0,
    fps: float = 2.0,
) -> list[ImageContent | TextContent]:
    """
    Capture a short video clip and ask Gemini to reason over it temporally.

    Useful for verifying motion: "Did the arm move down?",
    "Did the robot travel forward?", "Was an object picked up?"

    Args:
        prompt:     Question for Gemini about the clip.
        duration_s: Clip length in seconds.
        fps:        Frames per second.
    """
    try:
        result = cam_mod.capture_clip(duration_s, fps)
        analysis = vision_mod.analyze_clip(result["frames"], prompt)
        content: list[ImageContent | TextContent] = [
            TextContent(
                type="text",
                text=f"[{result['count']} frames, {duration_s}s]\n\n{analysis}",
            )
        ]
        for frame_b64 in result["frames"]:
            content.append(_image_content(frame_b64))
        return content
    except Exception as exc:
        return [TextContent(type="text", text=f"ERROR: {exc}")]


@mcp.tool()
def verify_action(
    action_description: str,
    before_frame_b64: str,
    after_frame_b64: str,
) -> dict:
    """
    Ask Gemini to judge whether an action succeeded by comparing before/after frames.

    Typical usage:
        1. before = capture_image()  (save the base64)
        2. perform action
        3. after  = capture_image()  (save the base64)
        4. verify_action("robot grasped the red block", before_b64, after_b64)

    Returns:
        {"success": bool, "confidence": "high"|"medium"|"low", "explanation": str}
    """
    try:
        result = vision_mod.verify_action(
            before_frame_b64, after_frame_b64, action_description
        )
        return _ok(result)
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def get_robot_state() -> list[ImageContent | TextContent]:
    """
    One-shot state snapshot: all motor positions + a live camera frame.
    Call this before planning any sequence of actions.
    """
    try:
        positions = robot_mod.get_all_positions()
        frame_data = cam_mod.capture_still()
        summary = (
            f"Motor positions — "
            f"left_wheel: {positions['left_wheel']}°, "
            f"right_wheel: {positions['right_wheel']}°, "
            f"arm: {positions['arm']}°, "
            f"gripper: {positions['gripper']}°"
        )
        return [
            _image_content(frame_data["frame"]),
            TextContent(type="text", text=summary),
        ]
    except Exception as exc:
        return [TextContent(type="text", text=f"ERROR: {exc}")]


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
