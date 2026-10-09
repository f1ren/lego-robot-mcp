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

import logging

from mcp_robot import config
from mcp_robot.rpi_client import get_client

log = logging.getLogger(__name__)

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

# A full raise ends with the arm pressed into its top stop, and cutting power
# there springs it back down: the arm falls all the way whenever power is cut
# while it's moving down, but stays put when cut while it's still. So back off
# in speed mode, hold with release=False until the motor has stopped the arm,
# and only then let go (a run_for_degrees back-off overshot further, to ~16
# from 31, in testing). Spliced into scripts that import time and name the
# arm's Motor `arm`. See config.ARM_TOP_BACK_OFF_DEG.
_ARM_SETTLE_BELOW_TOP = """
settle_target = arm.get_position() - {back_off_deg}
arm.start(-{speed})
settle_t0 = time.monotonic()
while arm.get_position() > settle_target and time.monotonic() - settle_t0 < 2.0:
    pass
arm.release = False
arm.run_for_degrees(-1, speed={speed})  # hold here; a 0° move is a zero-length ramp
time.sleep({hold_seconds})
arm.coast()
"""

# move_arm's full raise: up into the top stop, then settle just below it.
# The arm motor is inverted, so *degrees* is positive here.
_RAISE_ARM_FULLY = """
import json
import time
from buildhat import Motor

arm = Motor({port!r})
start = arm.get_position()
arm.run_for_degrees({degrees}, speed={speed})
top = arm.get_position()
{settle}
end = arm.get_position()
print(json.dumps({{"start": start, "top": top, "end": end, "delta": end - start}}))
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

# release=False stops buildhat's default post-move behaviour of coasting the
# motor ~0.2s after run_for_degrees finishes (see Motor._run_positional_ramp
# in the buildhat library). Coasting drops all holding torque, so a gripper
# closed on an object goes limp almost immediately and the object's weight/
# vibration can back-drive the gear train open. Closing the gripper, raising
# the arm, holding, and releasing all run inside a single RPi script (one
# SSH round-trip, one process) so the HAT firmware is guaranteed to still be
# actively applying hold torque for the entire arm-raise + hold window —
# splitting this across multiple run_python calls would risk a gap between
# calls where nothing is driving the gripper motor at all. The gripper then
# switches to open-loop PWM instead of coasting, and lift_arm registers it
# with RPiClient.hold_motor so this and every later script's exit keeps it
# driven until release_gripper_hold().
# The arm only rises to carry height (config.ARM_CARRY_RAISE_DEG), in speed
# mode until it gets there: a position move that short undershoots under the
# load (a 30° move_arm raise once moved the arm 10°). It's let go while still
# moving up, so gravity stops it, and it stays there unpowered.
_GRASP_HOLD_AND_LIFT = """
import json
import time
from buildhat import Motor

gripper = Motor({gripper_port!r})
gripper.release = False
gripper_start = gripper.get_position()
gripper.run_for_degrees({gripper_degrees}, speed={gripper_speed})
gripper_closed = gripper.get_position()

arm = Motor({arm_port!r})
arm_start = arm.get_position()
arm_target = arm_start + {arm_degrees}
arm.start({arm_speed})
arm_t0 = time.monotonic()
while arm.get_position() < arm_target and time.monotonic() - arm_t0 < 3.0:
    pass
arm.coast()
arm_end = arm.get_position()

time.sleep({hold_seconds})

gripper.pwm({gripper_hold_pwm})
arm_settled = arm.get_position()
gripper_end = gripper.get_position()

print(json.dumps({{
    "gripper_start": gripper_start,
    "gripper_closed": gripper_closed,
    "gripper_end": gripper_end,
    "gripper_delta": gripper_closed - gripper_start,
    "arm_start": arm_start,
    "arm_end": arm_end,
    "arm_delta": arm_end - arm_start,
    "arm_settled": arm_settled,
}}))
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

_COAST_MOTOR = """
import json
from buildhat import Motor

m = Motor({port!r})
m.coast()
print(json.dumps({{"ok": True, "position": m.get_position()}}))
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
# between them — so the button is guaranteed released as soon as this script
# returns, well before any VLM validation begins. Degree-based (not
# time-based): press_degrees is computed host-side from the measured
# distance to the switch (see server._square_up_to_target and
# navigation.mm_to_wheel_degrees) — the same px->mm->wheel-degrees pipeline
# navigate_to() uses for its own drive distances.
#
# WARNING: never `del pair` — BuildHAT triggers firmware jitter on destruction.
pair = MotorPair({left_port!r}, {right_port!r})
start_left = pair._leftmotor.get_position()
start_right = pair._rightmotor.get_position()

# Press forward into the button
pair.run_for_degrees({press_degrees}, {left_press_speed}, {right_press_speed})

mid_left = pair._leftmotor.get_position()
mid_right = pair._rightmotor.get_position()

# Immediately release by driving backward
pair.run_for_degrees({release_degrees}, {left_release_speed}, {right_release_speed})

end_left = pair._leftmotor.get_position()
end_right = pair._rightmotor.get_position()
print(json.dumps({{
    "left": end_left, "right": end_right,
    "left_delta": end_left - start_left, "right_delta": end_right - start_right,
    "press_left_delta": mid_left - start_left, "press_right_delta": mid_right - start_right,
    "release_left_delta": end_left - mid_left, "release_right_delta": end_right - mid_right,
}}))
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
        "gripper_held": gripper_held(),
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
    if port == config.PORT_GRIPPER:
        # Any gripper move ends a carry hold: cut its power first, otherwise
        # the end of this very script would re-apply the closing PWM.
        release_gripper_hold()
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
    speed: int,
    press_degrees: int,
    release_degrees: int | None = None,
) -> dict:
    """
    Press and immediately release a button in one atomic RPi script.

    Both press and release execute inside a single run_python call so no
    host round-trip (and no VLM pause) separates them. The button is
    physically released as soon as the script returns.

    Distance-based (wheel-encoder degrees), not time-based — the caller
    measures the real remaining distance to the switch (see
    server._square_up_to_target and navigation.mm_to_wheel_degrees) and
    passes it in as press_degrees, the same way navigate_to converts
    measured distance to wheel-encoder degrees for its own driving.

    Args:
        speed:           Wheel speed (positive = forward into button).
        press_degrees:   Wheel-encoder degrees to drive forward (into button).
        release_degrees: Wheel-encoder degrees to drive back afterward.
                         Defaults to config.CLICK_RELEASE_FRACTION of
                         press_degrees — just enough to clear the switch;
                         no need to return all the way to the start position.
    """
    if release_degrees is None:
        release_degrees = int(round(press_degrees * config.CLICK_RELEASE_FRACTION))
    # Motor A (left wheel) is physically inverted — negate so positive=forward.
    left_press   = -speed   # forward
    right_press  =  speed
    left_release =  speed   # backward
    right_release = -speed
    total_degrees = abs(press_degrees) + abs(release_degrees)
    total_timeout = max(30, total_degrees // 10 + 10)
    log.info(
        "click_button: driving forward %d° (press) then backward %d° (release) at speed %d",
        press_degrees, release_degrees, speed,
    )
    result = get_client().run_python(
        _CLICK_BUTTON.format(
            left_port=config.PORT_LEFT_WHEEL,
            right_port=config.PORT_RIGHT_WHEEL,
            press_degrees=press_degrees,
            left_press_speed=left_press,
            right_press_speed=right_press,
            release_degrees=release_degrees,
            left_release_speed=left_release,
            right_release_speed=right_release,
        ),
        timeout=total_timeout,
    )
    log.info(
        "click_button: pressed forward (left_delta=%s right_delta=%s), "
        "released backward (left_delta=%s right_delta=%s)",
        result.get("press_left_delta"), result.get("press_right_delta"),
        result.get("release_left_delta"), result.get("release_right_delta"),
    )
    return result


# ── arm ───────────────────────────────────────────────────────────────────────

def move_arm(degrees: int, speed: int = config.DEFAULT_ARM_SPEED) -> dict:
    """
    Move the arm by *degrees*. Positive = down, negative = up.
    After any downward move, raises 17° to keep the gripper clear of the ground
    and maximise wheel normal force. A full raise (by the arm's whole
    ARM_DOWN_DEG - ARM_UP_DEG travel or more) then settles just below the
    top stop so the arm stays up once let go — see _ARM_SETTLE_BELOW_TOP.

    Args:
        degrees: How far to move. Positive = down, negative = up.
        speed:   Motor speed 1–100.
    """
    direction = "down" if degrees > 0 else "up"
    log.info("move_arm: moving arm %s %d° at speed %d", direction, abs(degrees), speed)
    if degrees <= -(config.ARM_DOWN_DEG - config.ARM_UP_DEG):
        result = get_client().run_python(
            _RAISE_ARM_FULLY.format(
                port=config.PORT_ARM,
                degrees=-degrees,  # motor is physically inverted
                speed=speed,
                settle=_arm_settle_below_top(),
            ),
            timeout=max(30, abs(degrees) // 10 + 10),
        )
        log.info("move_arm: arm raised fully (top=%s), settled at %s", result.get("top"), result.get("end"))
        return result
    result = move_motor(config.PORT_ARM, -degrees, speed)  # motor is physically inverted; negate so positive=down as documented
    if degrees > 0:
        move_motor(config.PORT_ARM, 17, speed)  # raise 17° so gripper clears the ground
        log.info("move_arm: arm lowered %d° (delta=%s), raised 17° for ground clearance", abs(degrees), result.get("delta"))
    else:
        log.info("move_arm: arm raised %d° (delta=%s)", abs(degrees), result.get("delta"))
    return result


def _arm_settle_below_top() -> str:
    return _ARM_SETTLE_BELOW_TOP.format(
        back_off_deg=config.ARM_TOP_BACK_OFF_DEG,
        speed=config.ARM_TOP_BACK_OFF_SPEED,
        hold_seconds=config.ARM_TOP_HOLD_SECONDS,
    )


def lower_arm(speed: int = config.DEFAULT_ARM_SPEED) -> dict:
    """Lower arm fully to ground level, then raise 17° for wheel clearance."""
    return move_arm(config.ARM_DOWN_DEG, speed)


def lift_arm(speed: int = config.LIFT_ARM_SPEED) -> dict:
    """Close the gripper with holding torque, raise the arm
    config.ARM_CARRY_RAISE_DEG from lower_arm's pose to carry height (about
    halfway up, so the front camera still sees ahead over the object), hold
    the grip for config.LIFT_ARM_HOLD_SECONDS, and leave the gripper powered
    (config.GRIPPER_HOLD_PWM) so the object can be carried and dropped. The
    arm stays at carry height unpowered. raise_arm_fully takes it the rest
    of the way at the drop-off.
    The gripper stays powered across later tool calls until the gripper is
    next moved (control_gripper open, put), which cuts the power just before
    opening — see release_gripper_hold(). The close + raise run as a single
    RPi script (see _GRASP_HOLD_AND_LIFT) so the gripper stays under active
    hold for the whole arm-raise + hold window instead of coasting the
    instant the close finishes. Named to match the PDDL domain's lift-arm
    action (pddl/robot_domain.pddl), whose precondition already requires
    (holding ?o) — re-closing here re-affirms the grip immediately before
    the lift rather than trusting the earlier grasp to have held.

    Gripper close speed is fixed at config.DEFAULT_GRIPPER_SPEED; *speed*
    only controls the arm.
    """
    global _gripper_state
    arm_deg = config.ARM_CARRY_RAISE_DEG
    gripper_deg = config.GRIPPER_CLOSED_DEG
    client = get_client()
    # Registered before the script runs: its own exit must already keep the
    # gripper powered. An earlier hold is deliberately not released first —
    # the script's close takes the motor over directly, with no unpowered gap.
    client.hold_motor(config.PORT_GRIPPER, config.GRIPPER_HOLD_PWM)
    try:
        result = client.run_python(
            _GRASP_HOLD_AND_LIFT.format(
                gripper_port=config.PORT_GRIPPER,
                gripper_degrees=gripper_deg,
                gripper_speed=config.DEFAULT_GRIPPER_SPEED,
                arm_port=config.PORT_ARM,
                arm_degrees=arm_deg,
                arm_speed=speed,
                hold_seconds=config.LIFT_ARM_HOLD_SECONDS,
                gripper_hold_pwm=config.GRIPPER_HOLD_PWM,
            ),
            timeout=max(30, gripper_deg // 10 + arm_deg // 10 + int(config.LIFT_ARM_HOLD_SECONDS) + 18),
        )
    except Exception:
        # Don't leave a stalled gripper powered after a failed lift.
        try:
            release_gripper_hold()
        except Exception:
            log.exception("lift_arm: failed to release gripper hold after error")
        raise
    _gripper_state = "close"
    log.info(
        "lift_arm: closed gripper (delta=%s), raised arm to carry height (delta=%s of %s, "
        "settled at %s) and held %ss, gripper left powered at PWM %s until next opened",
        result.get("gripper_delta"), result.get("arm_delta"), arm_deg,
        result.get("arm_settled"), config.LIFT_ARM_HOLD_SECONDS, config.GRIPPER_HOLD_PWM,
    )
    return result


def raise_arm_fully(speed: int = config.LIFT_ARM_SPEED) -> dict:
    """Raise the arm from lift_arm's carry height into its top stop, then
    settle just below it: the release height for dropping a carried object
    into a bin. It's a full move_arm raise, so the arm stays up once let go
    (see _ARM_SETTLE_BELOW_TOP). Only the arm moves, so lift_arm's gripper
    hold stays on until the gripper is next moved."""
    return move_arm(-(config.ARM_DOWN_DEG - config.ARM_UP_DEG), speed)


def gripper_held() -> bool:
    """True while lift_arm's carry hold keeps the gripper motor powered."""
    return get_client().is_held(config.PORT_GRIPPER)


def release_gripper_hold() -> dict:
    """Cut the gripper's carry-hold power (fingers stay where they are; the
    held object is then free to push them open). No-op if not holding."""
    client = get_client()
    if not client.is_held(config.PORT_GRIPPER):
        return {"ok": True, "note": "gripper not held"}
    client.release_motor(config.PORT_GRIPPER)
    result = client.run_python(_COAST_MOTOR.format(port=config.PORT_GRIPPER))
    log.info("release_gripper_hold: gripper power cut (position=%s)", result.get("position"))
    return result


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
        log.info("control_gripper: gripper already %s, skipping", action)
        return {"action": action, "delta": 0, "note": "already at target"}

    verb = "opening" if action == "open" else "closing"
    log.info("control_gripper: %s gripper", verb)
    # Negative = opening direction, positive = closing direction (matches the
    # existing motor wiring convention used when the absolute approach worked).
    degrees = -config.GRIPPER_OPEN_DEG if action == "open" else config.GRIPPER_CLOSED_DEG
    result = move_motor(config.PORT_GRIPPER, degrees, speed)
    if action == "open":
        move_motor(config.PORT_GRIPPER, 17, speed)  # close 17° to release pressure from wheels
    result["action"] = action
    _gripper_state = action
    log.info("control_gripper: gripper %sd (delta=%s)", action, result.get("delta"))
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


def prep_for_press(speed: int = config.DEFAULT_GRIPPER_SPEED) -> dict:
    """Lift arm fully then close gripper — prep for a button/switch press.

    Raises the arm directly via move_arm rather than lift_arm: lift_arm now
    closes and holds the gripper *before* raising the arm (for carrying a
    grasped object), which is the opposite order this action needs (raise
    bare arm, then close into a fist for pressing) and would also add a
    pointless config.LIFT_ARM_HOLD_SECONDS wait.
    """
    arm_deg        = config.ARM_DOWN_DEG - config.ARM_UP_DEG
    arm_result     = move_arm(-arm_deg, speed=config.DEFAULT_ARM_SPEED)
    gripper_result = control_gripper("close", speed=speed)
    return {
        "action":  "prep_for_press",
        "arm":     arm_result,
        "gripper": gripper_result,
    }


def restore_grasp_pose(speed: int = config.DEFAULT_GRIPPER_SPEED) -> dict:
    """Lower arm fully then open gripper — reverses prep_for_press once a
    button/switch press is done.

    Arm before gripper, matching open-gripper's arm-lowered precondition in
    the PDDL domain (pddl/robot_domain.pddl) — so this leaves
    (is-in-grasp-pose) true again, not just the two raw facts. Without this,
    a PDDL action modeled on this tool (e.g. toggle-lights) that doesn't
    itself retract is-in-grasp-pose would have the planner believe the robot
    is still grasp-ready when the real arm/gripper were left mid-press.
    """
    arm_result     = lower_arm(speed=config.DEFAULT_ARM_SPEED)
    gripper_result = control_gripper("open", speed=speed)
    return {
        "action":  "restore_grasp_pose",
        "arm":     arm_result,
        "gripper": gripper_result,
    }
