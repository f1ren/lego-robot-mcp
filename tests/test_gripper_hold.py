"""
Gripper carry hold: lift_arm leaves the gripper powered across later tool
calls, and any gripper move cuts that power first.

Offline — RPiClient's SSH layer is replaced by a recorder, so these check the
bookkeeping and the generated script, not the BuildHAT itself.
"""
from __future__ import annotations

import pytest

from mcp_robot import config, robot, rpi_client


class _RecordingClient(rpi_client.RPiClient):
    """Real hold bookkeeping + script wrapping; records scripts instead of running them."""

    def __init__(self):
        super().__init__(host="test", user="test")
        self.scripts: list[str] = []

    def _run_python_once(self, script, timeout):
        self.scripts.append(self._build_script(script))
        return {"ok": True, "position": 0, "gripper_delta": 90, "arm_delta": 90}


@pytest.fixture
def client(monkeypatch):
    c = _RecordingClient()
    monkeypatch.setattr(robot, "get_client", lambda: c)
    monkeypatch.setattr(robot, "_gripper_state", None)
    return c


GRIPPER_IDX = ord(config.PORT_GRIPPER) - ord("A")


def test_no_preamble_when_nothing_held(client):
    robot.get_all_positions()
    assert "_BH_HELD" not in client.scripts[-1]


def test_lift_arm_keeps_gripper_held_for_later_scripts(client):
    robot.lift_arm()
    assert client.is_held(config.PORT_GRIPPER)
    # The lift script itself already exits with the hold kept…
    assert f"_BH_HELD = {{{GRIPPER_IDX}: {config.GRIPPER_HOLD_PWM}}}" in client.scripts[-1]
    assert f"gripper.pwm({config.GRIPPER_HOLD_PWM})" in client.scripts[-1]
    # …and so does an unrelated later call (e.g. a drive).
    robot.drive(20, 20, 1.0)
    assert "_BH_HELD" in client.scripts[-1]


def test_open_cuts_power_before_moving_fingers(client):
    robot.lift_arm()
    n = len(client.scripts)
    robot.control_gripper("open")
    coast, first_move = client.scripts[n], client.scripts[n + 1]
    assert "m.coast()" in coast and "_BH_HELD" not in coast
    assert "run_for_degrees" in first_move and "_BH_HELD" not in first_move
    assert not client.is_held(config.PORT_GRIPPER)


def test_put_releases_hold(client):
    robot.lift_arm()
    robot.put()
    assert not client.is_held(config.PORT_GRIPPER)
    assert all("_BH_HELD" not in s for s in client.scripts[-3:])


def test_raise_arm_fully_keeps_hold(client):
    robot.lift_arm()
    robot.raise_arm_fully()
    assert client.is_held(config.PORT_GRIPPER)
    script = client.scripts[-1]
    assert f"_BH_HELD = {{{GRIPPER_IDX}: {config.GRIPPER_HOLD_PWM}}}" in script
    assert f"Motor({config.PORT_GRIPPER!r})" not in script  # only the arm moves


def test_close_while_held_keeps_hold(client):
    robot.lift_arm()
    robot.control_gripper("close")  # already closed -> skipped
    assert client.is_held(config.PORT_GRIPPER)


def test_failed_lift_releases_hold(client, monkeypatch):
    calls = []

    def _fail_once(script, timeout):
        calls.append(script)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return {"ok": True, "position": 0}

    monkeypatch.setattr(client, "_run_python_once", _fail_once)
    with pytest.raises(RuntimeError):
        robot.lift_arm()
    assert not client.is_held(config.PORT_GRIPPER)
    assert "m.coast()" in calls[-1]


def test_wrapped_script_compiles_with_preamble(client):
    client.hold_motor(config.PORT_GRIPPER, 0.4)
    compile(client._build_script("print(1)"), "<rpi>", "exec")
