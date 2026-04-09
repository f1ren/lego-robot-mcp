"""
Robot motor control — primitives and high-level actions.

Motor layout (configure via env vars if different):
  A = left wheel
  B = right wheel
  C = arm  (up/down)
  D = gripper (open/close)

All functions return a dict with at least {"ok": bool}.
On error they raise RuntimeError (caught and wrapped by the MCP server).
"""
from __future__ import annotations

from mcp_robot import config
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
import json, threading
from buildhat import Motor

left  = Motor({left_port!r})
right = Motor({right_port!r})
results = {{}}

def move(motor, speed, seconds, key):
    motor.run_for_seconds(seconds, speed=speed)
    results[key] = motor.get_position()

t1 = threading.Thread(target=move, args=(left,  {left_speed},  {duration}, 'left'))
t2 = threading.Thread(target=move, args=(right, {right_speed}, {duration}, 'right'))
t1.start(); t2.start()
t1.join();  t2.join()
print(json.dumps(results))
"""

_STOP_WHEELS = """
import json
from buildhat import Motor

left  = Motor({left_port!r})
right = Motor({right_port!r})
left.stop()
right.stop()
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
    return {
        "left_wheel":  raw.get(config.PORT_LEFT_WHEEL),
        "right_wheel": raw.get(config.PORT_RIGHT_WHEEL),
        "arm":         raw.get(config.PORT_ARM),
        "gripper":     raw.get(config.PORT_GRIPPER),
        "ports":       raw,
    }


def move_motor(port: str, degrees: int, speed: int) -> dict:
    """Move a single motor by *degrees* at *speed*. Returns start/end positions."""
    result = get_client().run_python(
        _MOVE_SINGLE_MOTOR.format(port=port, degrees=degrees, speed=speed),
        timeout=max(30, abs(degrees) // 10 + 5),
    )
    return result


# ── wheel driving ─────────────────────────────────────────────────────────────

_WHEEL_DIRECTIONS = {
    # (left_speed_sign, right_speed_sign)
    "forward":  ( 1,  1),
    "backward": (-1, -1),
    "left":     (-1,  1),
    "right":    ( 1, -1),
}


def drive(
    direction: str,
    duration_s: float = 1.0,
    speed: int = config.DEFAULT_WHEEL_SPEED,
) -> dict:
    """
    Drive the robot wheels.

    Args:
        direction: "forward" | "backward" | "left" | "right" | "stop"
        duration_s: How long to run (seconds).
        speed: Motor speed 1–100.
    """
    if direction == "stop":
        return get_client().run_python(
            _STOP_WHEELS.format(
                left_port=config.PORT_LEFT_WHEEL,
                right_port=config.PORT_RIGHT_WHEEL,
            )
        )

    if direction not in _WHEEL_DIRECTIONS:
        raise ValueError(
            f"Unknown direction {direction!r}. "
            f"Use: {list(_WHEEL_DIRECTIONS)}"
        )

    ls, rs = _WHEEL_DIRECTIONS[direction]
    result = get_client().run_python(
        _DRIVE_WHEELS.format(
            left_port=config.PORT_LEFT_WHEEL,
            right_port=config.PORT_RIGHT_WHEEL,
            left_speed=ls * speed,
            right_speed=rs * speed,
            duration=duration_s,
        ),
        timeout=int(duration_s + 10),
    )
    result["direction"] = direction
    return result


# ── arm ───────────────────────────────────────────────────────────────────────

def move_arm(
    direction: str,
    degrees: int | None = None,
    speed: int = config.DEFAULT_ARM_SPEED,
) -> dict:
    """
    Move the arm.

    Args:
        direction: "up" or "down".
        degrees:   How far to move (default: full range from config).
        speed:     Motor speed 1–100.
    """
    if direction == "up":
        deg = -(degrees or abs(config.ARM_DOWN_DEG - config.ARM_UP_DEG))
    elif direction == "down":
        deg = degrees or abs(config.ARM_DOWN_DEG - config.ARM_UP_DEG)
    else:
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

    result = move_motor(config.PORT_ARM, deg, speed)
    result["direction"] = direction
    return result


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

def grasp(speed: int = config.DEFAULT_GRIPPER_SPEED) -> dict:
    """Lower arm then close gripper."""
    arm_result     = move_arm("down", speed=config.DEFAULT_ARM_SPEED)
    gripper_result = control_gripper("close", speed=speed)
    return {
        "action":  "grasp",
        "arm":     arm_result,
        "gripper": gripper_result,
    }


def put(speed: int = config.DEFAULT_GRIPPER_SPEED) -> dict:
    """Open gripper then raise arm."""
    gripper_result = control_gripper("open", speed=speed)
    arm_result     = move_arm("up", speed=config.DEFAULT_ARM_SPEED)
    return {
        "action":  "put",
        "gripper": gripper_result,
        "arm":     arm_result,
    }
