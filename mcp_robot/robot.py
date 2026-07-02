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
from buildhat import MotorPair

# MotorPair is required here — it is the only BuildHAT API that commands
# both wheel motors in a single firmware call, guaranteeing synchronised
# start and stop.  Replacing this with two separate Motor instances would
# cause each motor to start/stop independently, producing timing skew,
# unpredictable heading drift, and non-repeatable manoeuvres.
#
# WARNING: never `del pair` (or any Motor/MotorPair) — BuildHAT triggers a
# firmware jitter on destruction that makes motors twitch.  Read encoder
# positions via pair._leftmotor / pair._rightmotor instead of re-creating
# Motor objects after the move.
pair = MotorPair({left_port!r}, {right_port!r})
pair.run_for_seconds({duration}, {left_speed}, {right_speed})
print(json.dumps({{"left": pair._leftmotor.get_position(), "right": pair._rightmotor.get_position()}}))
"""

_STOP_WHEELS = """
import json
from buildhat import MotorPair

# MotorPair is required here — it is the only BuildHAT API that commands
# both wheel motors in a single firmware call, guaranteeing synchronised
# start and stop.  Replacing this with two separate Motor instances would
# cause each motor to start/stop independently, producing timing skew,
# unpredictable heading drift, and non-repeatable manoeuvres.
#
# WARNING: never `del pair` (or any Motor/MotorPair) — BuildHAT triggers a
# firmware jitter on destruction that makes motors twitch.  Read encoder
# positions via pair._leftmotor / pair._rightmotor instead of re-creating
# Motor objects after the move.
pair = MotorPair({left_port!r}, {right_port!r})
pair.stop()
print(json.dumps({{"ok": True, "left": pair._leftmotor.get_position(), "right": pair._rightmotor.get_position()}}))
"""

_DRIVE_WHEELS_BY_DEGREES = """
import json
from buildhat import MotorPair

# MotorPair is required here — it is the only BuildHAT API that commands
# both wheel motors in a single firmware call, guaranteeing synchronised
# start and stop.  Replacing this with two separate Motor instances would
# cause each motor to start/stop independently, producing timing skew,
# unpredictable heading drift, and non-repeatable manoeuvres.  This is
# especially critical for angle-based driving where precision matters.
#
# WARNING: never `del pair` (or any Motor/MotorPair) — BuildHAT triggers a
# firmware jitter on destruction that makes motors twitch.  Read encoder
# positions via pair._leftmotor / pair._rightmotor instead of re-creating
# Motor objects after the move.
pair = MotorPair({left_port!r}, {right_port!r})
pair.run_for_degrees({degrees}, {left_speed}, {right_speed})
print(json.dumps({{"left": pair._leftmotor.get_position(), "right": pair._rightmotor.get_position()}}))
"""

_CLICK_BUTTON = """
import json
from buildhat import MotorPair

# Both press and release run in a single RPi script — no host round-trip
# between them — so the button is guaranteed released within
# press_duration + release_duration seconds, well before any VLM validation begins.
#
# WARNING: never `del pair` — BuildHAT triggers firmware jitter on destruction.
pair = MotorPair({left_port!r}, {right_port!r})

# Press forward into the button
pair.run_for_seconds({press_duration}, {left_press_speed}, {right_press_speed})

# Immediately release by driving backward
pair.run_for_seconds({release_duration}, {left_release_speed}, {right_release_speed})

print(json.dumps({{"left": pair._leftmotor.get_position(), "right": pair._rightmotor.get_position()}}))
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
    return result


# ── wheel driving ─────────────────────────────────────────────────────────────

def drive(
    left_speed: int,
    right_speed: int,
    duration_s: float = 1.0,
) -> dict:
    """
    Drive the robot wheels. Positive speed = forward for both wheels.

    Args:
        left_speed:  Speed for the left wheel, -100 to 100. Positive = forward.
        right_speed: Speed for the right wheel, -100 to 100. Positive = forward.
        duration_s:  How long to run (seconds). Pass 0 to stop both wheels.
    """
    # Left motor (A) is physically inverted — negate so positive = forward matches right wheel convention.
    # Verified: MotorPair('A','B').run_for_seconds(1, -20, 20) moves forward.
    if duration_s == 0:
        return get_client().run_python(
            _STOP_WHEELS.format(
                left_port=config.PORT_LEFT_WHEEL,
                right_port=config.PORT_RIGHT_WHEEL,
            )
        )

    return get_client().run_python(
        _DRIVE_WHEELS.format(
            left_port=config.PORT_LEFT_WHEEL,
            right_port=config.PORT_RIGHT_WHEEL,
            left_speed=-left_speed,  # motor A is inverted; negate to keep positive=forward
            right_speed=right_speed,
            duration=duration_s,
        ),
        timeout=int(duration_s + 10),
    )


def drive_degrees(
    degrees: int,
    left_speed: int,
    right_speed: int,
) -> dict:
    """
    Drive both wheel motors by exactly *degrees* of encoder rotation.

    Unlike drive() which runs for a fixed duration, this runs until each
    wheel has physically rotated *degrees* encoder-degrees, giving
    repeatable distances and turns regardless of battery voltage or load.

    The direction of travel is controlled by the sign of the speed
    arguments, not the sign of *degrees* (pass abs values for degrees).
    For an in-place turn, set left_speed = -right_speed; for straight
    travel set both to the same positive value.

    Args:
        degrees:     Motor encoder degrees each wheel rotates (positive).
        left_speed:  Left wheel speed, -100 to 100. Positive = forward.
        right_speed: Right wheel speed, -100 to 100. Positive = forward.
    """
    return get_client().run_python(
        _DRIVE_WHEELS_BY_DEGREES.format(
            left_port=config.PORT_LEFT_WHEEL,
            right_port=config.PORT_RIGHT_WHEEL,
            degrees=abs(degrees),
            left_speed=-left_speed,  # motor A is physically inverted; negate to keep positive=forward
            right_speed=right_speed,
        ),
        timeout=max(30, abs(degrees) // 50 + 10),
    )


def turn(body_degrees: float, speed: int) -> dict:
    """
    Rotate the robot body by *body_degrees* in place (both wheels counter-rotating).

    Positive body_degrees = clockwise when viewed from above.
    Negative body_degrees = counter-clockwise when viewed from above.

    The encoder travel per wheel is computed from the configured wheel geometry:
        encoder_deg = abs(body_degrees) * TURN_ENCODER_DEG_PER_BODY_DEG
    """
    encoder_deg = int(abs(body_degrees) * config.TURN_ENCODER_DEG_PER_BODY_DEG)
    if body_degrees >= 0:
        # CW: left wheel forward, right wheel backward
        left_speed, right_speed = speed, -speed
    else:
        # CCW: left wheel backward, right wheel forward
        left_speed, right_speed = -speed, speed
    return drive_degrees(encoder_deg, left_speed, right_speed)


def click_button(
    speed: int = 20,
    press_duration_s: float = 1.0,
    release_duration_s: float = 1.0,
) -> dict:
    """
    Press and immediately release a button in one atomic RPi script.

    Both press and release execute inside a single run_python call so no
    host round-trip (and no VLM pause) separates them.  The button is
    physically released within press_duration_s + release_duration_s seconds.

    Args:
        speed:              Wheel speed (positive = forward into button).
        press_duration_s:   Time driving forward to depress the button.
        release_duration_s: Time driving backward to un-press the button.
    """
    # Motor A (left wheel) is physically inverted — negate so positive=forward.
    left_press   = -speed   # forward
    right_press  =  speed
    left_release =  speed   # backward
    right_release = -speed
    total_timeout = int(press_duration_s + release_duration_s + 10)
    result = get_client().run_python(
        _CLICK_BUTTON.format(
            left_port=config.PORT_LEFT_WHEEL,
            right_port=config.PORT_RIGHT_WHEEL,
            press_duration=press_duration_s,
            left_press_speed=left_press,
            right_press_speed=right_press,
            release_duration=release_duration_s,
            left_release_speed=left_release,
            right_release_speed=right_release,
        ),
        timeout=total_timeout,
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
    After any downward move, raises 17° to keep the gripper clear of the ground
    and maximise wheel normal force.

    Args:
        degrees: How far to move. Positive = down, negative = up.
        speed:   Motor speed 1–100.
    """
    result = move_motor(config.PORT_ARM, -degrees, speed)  # motor is physically inverted; negate so positive=down as documented
    if degrees > 0:
        move_motor(config.PORT_ARM, 17, speed)  # raise 17° so gripper clears the ground
    return result


def lower_arm(speed: int = config.DEFAULT_ARM_SPEED) -> dict:
    """Lower arm fully to ground level, then raise 17° for wheel clearance."""
    return move_arm(config.ARM_DOWN_DEG, speed)


# ── gripper ───────────────────────────────────────────────────────────────────

# Last known gripper state.  None = unknown (first call always moves).
# LEGO motors have incremental encoders that reset on power-cycle, so absolute
# position targets are unreliable.  We use fixed relative travel and track
# logical state instead.
_gripper_state: str | None = None


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
    global _gripper_state

    if action not in ("open", "close"):
        raise ValueError(f"action must be 'open' or 'close', got {action!r}")

    if _gripper_state == action:
        return {"action": action, "delta": 0, "note": "already at target"}

    # Negative = opening direction, positive = closing direction (matches the
    # existing motor wiring convention used when the absolute approach worked).
    degrees = -config.GRIPPER_OPEN_DEG if action == "open" else config.GRIPPER_CLOSED_DEG
    result = move_motor(config.PORT_GRIPPER, degrees, speed)
    if action == "open":
        move_motor(config.PORT_GRIPPER, 17, speed)  # close 17° to release pressure from wheels
    result["action"] = action
    _gripper_state = action
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
