"""
Full arm raises settle just below the top stop before letting the arm go:
back off in speed mode, hold with release=False, then coast — all in the
raise's own script. Partial raises (lower_arm's 17° clearance raise, and
lift_arm's raise to carry height) don't back off.

Offline — RPiClient's SSH layer is replaced by a recorder, so these check the
generated scripts, not the BuildHAT itself.
"""
from __future__ import annotations

import pytest

from mcp_robot import config, robot, rpi_client

FULL = config.ARM_DOWN_DEG - config.ARM_UP_DEG
SETTLE = (
    f"arm.start(-{config.ARM_TOP_BACK_OFF_SPEED})",
    "arm.release = False",
    f"arm.run_for_degrees(-1, speed={config.ARM_TOP_BACK_OFF_SPEED})",
    f"time.sleep({config.ARM_TOP_HOLD_SECONDS})",
    "arm.coast()",
)


class _RecordingClient(rpi_client.RPiClient):
    """Real script wrapping; records scripts instead of running them."""

    def __init__(self):
        super().__init__(host="test", user="test")
        self.scripts: list[str] = []

    def _run_python_once(self, script, timeout):
        self.scripts.append(self._build_script(script))
        return {"ok": True, "position": 0}


@pytest.fixture
def client(monkeypatch):
    c = _RecordingClient()
    monkeypatch.setattr(robot, "get_client", lambda: c)
    monkeypatch.setattr(robot, "_gripper_state", None)
    return c


def _assert_in_order(script: str, *needles: str) -> None:
    pos = -1
    for needle in needles:
        found = script.find(needle, pos + 1)
        assert found > pos, f"{needle!r} missing or out of order"
        pos = found


def _settles(script: str) -> bool:
    return "settle_target" in script


def test_full_raise_settles_below_top_in_the_same_script(client):
    robot.move_arm(-FULL, speed=15)
    assert len(client.scripts) == 1
    _assert_in_order(client.scripts[0], f"arm.run_for_degrees({FULL}, speed=15)", *SETTLE)
    compile(client.scripts[0], "<rpi>", "exec")


def test_partial_raise_does_not_back_off(client):
    robot.move_arm(-30, speed=15)
    assert len(client.scripts) == 1
    assert "m.run_for_degrees(30, speed=15)" in client.scripts[0]
    assert not _settles(client.scripts[0])


def test_lower_arm_clearance_raise_does_not_back_off(client):
    robot.lower_arm(speed=15)
    assert len(client.scripts) == 2  # lower, then the 17° clearance raise
    assert not any(_settles(s) for s in client.scripts)


def test_lift_arm_raises_only_to_carry_height(client):
    robot.lift_arm(speed=15)
    script = client.scripts[-1]
    _assert_in_order(
        script,
        "gripper.run_for_degrees(",
        f"arm_target = arm_start + {config.ARM_CARRY_RAISE_DEG}",
        "arm.start(15)",
        "while arm.get_position() < arm_target",
        "arm.coast()",  # let go while still moving up
        f"time.sleep({config.LIFT_ARM_HOLD_SECONDS})",
        f"gripper.pwm({config.GRIPPER_HOLD_PWM})",
        "arm_settled = arm.get_position()",
    )
    assert f"arm.run_for_degrees({FULL}" not in script
    assert not _settles(script)
    assert script.count("arm.coast()") == 1
    compile(script, "<rpi>", "exec")  # with the gripper hold preamble


@pytest.mark.parametrize("action", [robot.put, robot.prep_for_press, robot.raise_arm_fully])
def test_compound_full_raises_settle(client, action):
    action()
    raises = [s for s in client.scripts if f"arm.run_for_degrees({FULL}" in s]
    assert len(raises) == 1 and _settles(raises[0])
