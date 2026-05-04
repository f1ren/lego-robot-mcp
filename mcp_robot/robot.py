"""
Robot motor control — primitives and high-level actions.

Motor layout (configure via env vars if different):
  A = left wheel
  B = right wheel
  C = gripper (open/close)
  D = arm  (up/down)

All functions return a dict with at least {"ok": bool}.
On error they raise RuntimeError (caught and wrapped by the MCP server).
"""
from __future__ import annotations

from mcp_robot import config, viz
from mcp_robot.rpi_client import get_client

# ── RPi script templates ──────────────────────────────────────────────────────

_GET_ALL_POSITIONS = """
import json
from buildhat import Motor

positions = {{}}
for port in {ports!r}:
    try:
        positions[port] = Motor(port).get_position()
    except Exception as e:
        positions[port] = {{"error": str(e)}}
print(json.dumps(positions))
"""

_MOVE_SINGLE_MOTOR = """
import json
from buildhat import Motor

m = Motor({port!r})
start = m.get_position()
m.run_for_degrees({degrees}, speed={speed})
end = m.get_position()
print(json.dumps({{"start": start, "end": end, "delta": end - start}}))
"""

_DRIVE_WHEELS = """
import json
from buildhat import Motor, MotorPair

pair = MotorPair({left_port!r}, {right_port!r})
pair.run_for_seconds({duration}, {left_speed}, {right_speed})
del pair
left  = Motor({left_port!r})
right = Motor({right_port!r})
print(json.dumps({{"left": left.get_position(), "right": right.get_position()}}))
"""

_STOP_WHEELS = """
import json
from buildhat import Motor, MotorPair

pair = MotorPair({left_port!r}, {right_port!r})
pair.stop()
del pair
left  = Motor({left_port!r})
right = Motor({right_port!r})
print(json.dumps({{"ok": True, "left": left.get_position(), "right": right.get_position()}}))
"""


# ── primitives ────────────────────────────────────────────────────────────────

def get_all_positions() -> dict:
    """Return current position (degrees) for all four motor ports."""
    ports = [
        config.PORT_LEFT_WHEEL,
        config.PORT_RIGHT_WHEEL,
        config.PORT_ARM,
        config.PORT_GRIPPER,
    ]
    raw = get_client().run_python(_GET_ALL_POSITIONS.format(ports=ports))
    positions = {
        "left_wheel":  raw.get(config.PORT_LEFT_WHEEL),
        "right_wheel": raw.get(config.PORT_RIGHT_WHEEL),
        "arm":         raw.get(config.PORT_ARM),
        "gripper":     raw.get(config.PORT_GRIPPER),
        "ports":       raw,
    }
    viz.log_motor_positions(positions)
    return positions


_PORT_TO_NAME = {
    config.PORT_LEFT_WHEEL:  "left_wheel",
    config.PORT_RIGHT_WHEEL: "right_wheel",
    config.PORT_ARM:         "arm",
    config.PORT_GRIPPER:     "gripper",
}


def move_motor(port: str, degrees: int, speed: int) -> dict:
    """Move a single motor by *degrees* at *speed*. Returns start/end positions."""
    result = get_client().run_python(
        _MOVE_SINGLE_MOTOR.format(port=port, degrees=degrees, speed=speed),
        timeout=max(30, abs(degrees) // 10 + 5),
    )
    name = _PORT_TO_NAME.get(port, port)
    viz.log_motor_positions({name: result["start"]})
    viz.log_motor_positions({name: result["end"]})
    return result


# ── wheel driving ─────────────────────────────────────────────────────────────

def drive(
    left_speed: int,
    right_speed: int,
    duration_s: float = 1.0,
) -> dict:
    """
    Drive the robot wheels directly.

    Args:
        left_speed:  Speed for the left wheel, -100 to 100. Positive/negative
                     direction must be determined empirically.
        right_speed: Speed for the right wheel, -100 to 100.
        duration_s:  How long to run (seconds). Pass 0 to stop both wheels.
    """
    if duration_s == 0:
        result = get_client().run_python(
            _STOP_WHEELS.format(
                left_port=config.PORT_LEFT_WHEEL,
                right_port=config.PORT_RIGHT_WHEEL,
            )
        )
        viz.log_motor_positions({
            "left_wheel":  result.get("left"),
            "right_wheel": result.get("right"),
        })
        return result

    result = get_client().run_python(
        _DRIVE_WHEELS.format(
            left_port=config.PORT_LEFT_WHEEL,
            right_port=config.PORT_RIGHT_WHEEL,
            left_speed=left_speed,
            right_speed=right_speed,
            duration=duration_s,
        ),
        timeout=int(duration_s + 10),
    )
    viz.log_motor_positions({
        "left_wheel":  result.get("left"),
        "right_wheel": result.get("right"),
    })
    return result


# ── arm ───────────────────────────────────────────────────────────────────────

def move_arm(degrees: int, speed: int = config.DEFAULT_ARM_SPEED) -> dict:
    """
    Move the arm by *degrees*. Positive = down, negative = up.

    Args:
        degrees: How far to move. Positive = down, negative = up.
        speed:   Motor speed 1–100.
    """
    return move_motor(config.PORT_ARM, -degrees, speed)  # motor is physically inverted; negate so positive=down as documented


# ── gripper ───────────────────────────────────────────────────────────────────

def control_gripper(
    action: str,
    speed: int = config.DEFAULT_GRIPPER_SPEED,
) -> dict:
    """
    Open or close the gripper.

    Args:
        action: "open" or "close".
        speed:  Motor speed 1–100.
    """
    if action not in ("open", "close"):
        raise ValueError(f"action must be 'open' or 'close', got {action!r}")

    current = get_client().run_python(
        _GET_ALL_POSITIONS.format(ports=[config.PORT_GRIPPER])
    )
    current_pos = current.get(config.PORT_GRIPPER, 0)
    if isinstance(current_pos, dict):
        raise RuntimeError(f"Gripper motor error: {current_pos.get('error')}")

    target = config.GRIPPER_OPEN_DEG if action == "open" else config.GRIPPER_CLOSED_DEG
    delta = target - current_pos

    if abs(delta) < 3:
        return {"action": action, "start": current_pos, "end": current_pos, "delta": 0, "note": "already at target"}

    result = move_motor(config.PORT_GRIPPER, delta, speed)
    result["action"] = action
    return result


# ── high-level compound actions ───────────────────────────────────────────────


def put(speed: int = config.DEFAULT_GRIPPER_SPEED) -> dict:
    """Open gripper then raise arm."""
    arm_deg        = config.ARM_DOWN_DEG - config.ARM_UP_DEG
    gripper_result = control_gripper("open", speed=speed)
    arm_result     = move_arm(-arm_deg, speed=config.DEFAULT_ARM_SPEED)
    return {
        "action":  "put",
        "gripper": gripper_result,
        "arm":     arm_result,
    }
