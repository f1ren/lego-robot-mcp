"""
Unit tests for mcp_robot.server.drive_to's long-range overshoot guard.

Mocks the hardware/vision boundary directly on mcp_robot.server so drive_to's
branching logic (single drive vs. 2-drive auto-refine) can be exercised
offline and deterministically, without a camera, VQA, or the robot:

  _measure_target         -> canned (angle_deg, distance_mm) readings, in
                              call order (or a raised exception)
  _target_too_far         -> canned guard string-or-None, in call order
  robot_mod.drive_degrees -> records (degrees, left_speed, right_speed)
                              calls, returns a canned encoder-position dict
  _with_change_analysis   -> runs action_fn() (so the drive_degrees mock
                              above still records its call) then returns a
                              canned ok/change_description dict, skipping
                              camera capture, VQA, and the recorder entirely
"""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from mcp_robot import config
from mcp_robot import navigation as nav_mod
from mcp_robot import server

# Derived from the live config so these tests track DRIVE_TO_LONG_RANGE_BODY_LENGTHS/
# DRIVE_TO_PARTIAL_FRACTION/DRIVE_TO_TOUCH_OFFSET_MM if they're ever retuned (as they
# already have been once).
_THRESHOLD_MM = config.DRIVE_TO_LONG_RANGE_BODY_LENGTHS * config.ROBOT_BODY_LENGTH_MM
_SHORT_MM = _THRESHOLD_MM * 0.5
_LONG_MM = _THRESHOLD_MM * 1.5
_OFFSET_MM = config.DRIVE_TO_TOUCH_OFFSET_MM


def _wheel_deg(mm: float) -> int:
    return int(round(nav_mod.mm_to_wheel_degrees(mm)))


def _first_leg_mm(raw_mm: float) -> float:
    """Mirrors drive_to()'s long-range first-leg formula: the partial
    fraction applies to the touch-adjusted distance, not the raw one."""
    return max(0.0, raw_mm - _OFFSET_MM) * config.DRIVE_TO_PARTIAL_FRACTION


def _patch_measure_and_guard(monkeypatch, measurements, guards):
    """measurements: list of (angle_deg, distance_mm) tuples (or exceptions),
    consumed in call order by _measure_target. guards: list of str|None (or
    exceptions), consumed in call order by _target_too_far."""
    monkeypatch.setattr(server, "_measure_target", Mock(side_effect=measurements))
    monkeypatch.setattr(server, "_target_too_far", Mock(side_effect=guards))


def _patch_change_analysis(monkeypatch, results):
    """results: list of dicts, each either {"ok": False, "error": "..."} or
    {"change_description": "..."} (ok defaults True), consumed in call order.
    Returns the list of captured call kwargs for assertions."""
    calls = []
    queue = list(results)

    def _fake(action_desc, expected, action_fn, **kwargs):
        calls.append({"action_desc": action_desc, "expected": expected, **kwargs})
        action_result = action_fn()
        if not queue:
            raise AssertionError("_with_change_analysis called more times than the test expected")
        canned = queue.pop(0)
        if canned.get("ok", True) is False:
            return {"ok": False, "error": canned.get("error", "mock failure")}
        return {"ok": True, **action_result, "change_description": canned.get("change_description", "moved")}

    monkeypatch.setattr(server, "_with_change_analysis", _fake)
    return calls


def _patch_drive_degrees(monkeypatch, returns=None):
    mock_drive = Mock(side_effect=returns) if returns else Mock(return_value={"left": 111, "right": 222})
    monkeypatch.setattr(server.robot_mod, "drive_degrees", mock_drive)
    return mock_drive


def _drive_to(**overrides):
    kwargs = {"target_class_yolo": "cup", "target_class_free_text": "a red cup", "speed": 20}
    kwargs.update(overrides)
    return server.drive_to(**kwargs)


# ── Case 1: short range — single drive, no auto-refine ───────────────────────

def test_short_range_single_drive(monkeypatch):
    # _SHORT_MM is centroid-to-centroid; the actual drive is that minus the
    # touch offset (see test_short_range_drive_subtracts_touch_offset below
    # for a case with a larger, less coincidentally-small margin).
    driven_mm = _SHORT_MM - _OFFSET_MM
    assert driven_mm > 0, "test fixture assumes _SHORT_MM exceeds the touch offset"

    _patch_measure_and_guard(monkeypatch, measurements=[(5.0, _SHORT_MM)], guards=[None])
    drive_mock = _patch_drive_degrees(monkeypatch)
    calls = _patch_change_analysis(monkeypatch, [{"change_description": "drove to cup"}])

    result = _drive_to()

    assert result["ok"] is True
    assert result["measured_angle_deg"] == 5.0
    assert result["measured_distance_mm"] == _SHORT_MM
    assert result["change_description"] == "drove to cup"
    assert "drives_executed" not in result
    assert "first_drive" not in result
    assert "partial_drive" not in result
    assert "driven_distance_mm" not in result
    assert len(calls) == 1
    drive_mock.assert_called_once_with(_wheel_deg(driven_mm), 20, 20)


def test_short_range_drive_subtracts_touch_offset(monkeypatch):
    """Larger, round-number case for the touch-offset subtraction, so the
    expected drive distance isn't coincidentally close to zero."""
    raw_mm = 250.0
    assert raw_mm <= _THRESHOLD_MM, "test fixture assumes this stays in the short-range branch"
    driven_mm = raw_mm - _OFFSET_MM

    _patch_measure_and_guard(monkeypatch, measurements=[(0.0, raw_mm)], guards=[None])
    drive_mock = _patch_drive_degrees(monkeypatch)
    _patch_change_analysis(monkeypatch, [{"change_description": "drove to cup"}])

    result = _drive_to()

    assert result["ok"] is True
    assert result["measured_distance_mm"] == raw_mm
    drive_mock.assert_called_once_with(_wheel_deg(driven_mm), 20, 20)


def test_distance_within_touch_offset_skips_drive(monkeypatch):
    """A nonzero but sub-offset centroid-to-centroid distance (unlike the
    near-zero case in test_already_at_target_before_first_drive) should also
    be treated as already touching — not driven into."""
    raw_mm = _OFFSET_MM * 0.5
    _patch_measure_and_guard(monkeypatch, measurements=[(0.0, raw_mm)], guards=[None])
    drive_mock = _patch_drive_degrees(monkeypatch)
    calls = _patch_change_analysis(monkeypatch, [])

    result = _drive_to()

    assert result["ok"] is True
    assert "within touch offset" in result["message"]
    assert len(calls) == 0
    drive_mock.assert_not_called()


# ── Case 2: long range — auto-refines in exactly 2 drives ────────────────────

def test_long_range_auto_refines_in_two_drives(monkeypatch):
    # Remeasured second-leg distance, comfortably above the touch offset so
    # the second drive actually fires (see the dedicated within-offset case
    # below for when it doesn't).
    second_distance_mm = 200.0
    second_driven_mm = second_distance_mm - _OFFSET_MM
    assert second_driven_mm > 0, "test fixture assumes second_distance_mm exceeds the touch offset"

    _patch_measure_and_guard(
        monkeypatch,
        measurements=[(5.0, _LONG_MM), (2.0, second_distance_mm)],
        guards=[None, None],
    )
    drive_mock = _patch_drive_degrees(
        monkeypatch, returns=[{"left": 100, "right": 100}, {"left": 460, "right": 460}]
    )
    calls = _patch_change_analysis(monkeypatch, [
        {"change_description": "drove most of the way"},
        {"change_description": "arrived at cup"},
    ])

    result = _drive_to()

    first_leg_mm = _first_leg_mm(_LONG_MM)

    assert result["ok"] is True
    assert result["measured_angle_deg"] == 5.0            # from the *first* measurement
    assert result["measured_distance_mm"] == _LONG_MM     # original full distance, not the remainder
    assert result["driven_distance_mm"] == pytest.approx(first_leg_mm + second_driven_mm)
    assert result["drives_executed"] == 2
    assert result["change_description"] == "arrived at cup"   # final leg's own verification
    assert result["left"] == 460 and result["right"] == 460   # final leg's encoder readings
    assert result["first_drive"]["driven_mm"] == pytest.approx(first_leg_mm)
    assert result["first_drive"]["change_description"] == "drove most of the way"
    assert result["first_drive"]["left"] == 100 and result["first_drive"]["right"] == 100
    assert "auto-refined in 2 drives" in result["message"]
    assert len(calls) == 2

    assert drive_mock.call_args_list[0].args[0] == _wheel_deg(first_leg_mm)
    assert drive_mock.call_args_list[1].args[0] == _wheel_deg(second_driven_mm)


def test_long_range_first_leg_never_overshoots_touch_offset(monkeypatch):
    """Regression test for a distance just over the long-range threshold: the
    85% fraction must apply to the touch-adjusted distance, not the raw one.
    Applied to the raw distance, 400mm would drive 340mm (0.85 x 400) and
    leave only a 60mm centroid-gap — already less than the 130mm touch
    offset, i.e. leg 1 alone would overshoot into the touch zone before leg
    2's touch-aware logic ever runs."""
    raw_mm = 400.0
    breakeven_mm = _OFFSET_MM / (1 - config.DRIVE_TO_PARTIAL_FRACTION)
    assert _THRESHOLD_MM < raw_mm < breakeven_mm, (
        "test fixture assumes raw_mm is in the danger zone where the naive "
        "(pre-fix) formula would have overshot the touch offset on leg 1 alone"
    )
    first_leg_mm = _first_leg_mm(raw_mm)
    remaining_centroid_gap = raw_mm - first_leg_mm

    _patch_measure_and_guard(
        monkeypatch,
        measurements=[(0.0, raw_mm), (0.0, remaining_centroid_gap)],
        guards=[None, None],
    )
    drive_mock = _patch_drive_degrees(monkeypatch)
    _patch_change_analysis(monkeypatch, [
        {"change_description": "drove most of the way"},
        {"change_description": "arrived at cup"},
    ])

    result = _drive_to()

    assert result["ok"] is True
    assert remaining_centroid_gap >= _OFFSET_MM, (
        "the first leg alone must never drive past the touch offset"
    )
    assert drive_mock.call_args_list[0].args[0] == _wheel_deg(first_leg_mm)


# ── Edge cases around the auto-refine step ────────────────────────────────────

def test_first_drive_failure_skips_second_drive(monkeypatch):
    _patch_measure_and_guard(monkeypatch, measurements=[(5.0, _LONG_MM)], guards=[None])
    _patch_drive_degrees(monkeypatch)
    calls = _patch_change_analysis(monkeypatch, [{"ok": False, "error": "ssh timeout"}])

    result = _drive_to()

    assert result == {
        "ok": False,
        "error": "ssh timeout",
        "measured_angle_deg": 5.0,
        "measured_distance_mm": _LONG_MM,
    }
    assert "driven_distance_mm" not in result  # unknown how far a failed drive actually got
    assert len(calls) == 1  # no second drive attempted after a hardware failure


def test_second_measurement_unavailable_falls_back_to_first_drive(monkeypatch):
    _patch_measure_and_guard(
        monkeypatch,
        measurements=[(5.0, _LONG_MM), (None, None)],  # target lost after first drive
        guards=[None],
    )
    _patch_drive_degrees(monkeypatch)
    calls = _patch_change_analysis(monkeypatch, [{"change_description": "drove most of the way"}])

    result = _drive_to()

    assert result["ok"] is True
    assert result["partial_drive"] is True
    assert result["driven_distance_mm"] == pytest.approx(_first_leg_mm(_LONG_MM))
    assert "Target not visible for a second measurement" in result["message"]
    assert "call drive_to() again" in result["message"]
    assert len(calls) == 1


def test_second_measurement_error_falls_back_to_first_drive(monkeypatch):
    _patch_measure_and_guard(
        monkeypatch,
        measurements=[(5.0, _LONG_MM), RuntimeError("camera disconnected")],
        guards=[None],
    )
    _patch_drive_degrees(monkeypatch)
    calls = _patch_change_analysis(monkeypatch, [{"change_description": "drove most of the way"}])

    result = _drive_to()

    assert result["ok"] is True
    assert result["partial_drive"] is True
    assert "camera disconnected" in result["message"]
    assert len(calls) == 1


def test_second_guard_falls_back_to_first_drive(monkeypatch):
    guard_msg = "ERROR: Target is too far for manual drive/turn (distance=999px, threshold=100px)."
    _patch_measure_and_guard(
        monkeypatch,
        measurements=[(5.0, _LONG_MM), (2.0, 50.0)],
        guards=[None, guard_msg],
    )
    _patch_drive_degrees(monkeypatch)
    calls = _patch_change_analysis(monkeypatch, [{"change_description": "drove most of the way"}])

    result = _drive_to()

    assert result["ok"] is True
    assert result["partial_drive"] is True
    assert "too far for manual drive/turn" in result["message"]
    assert "needs navigate_to()" in result["message"]
    assert len(calls) == 1  # second drive never physically attempted


def test_already_at_target_before_first_drive(monkeypatch):
    _patch_measure_and_guard(monkeypatch, measurements=[(0.0, 0.1)], guards=[None])
    _patch_drive_degrees(monkeypatch)
    calls = _patch_change_analysis(monkeypatch, [])

    result = _drive_to()

    assert result["ok"] is True
    assert "within touch offset" in result["message"]
    assert len(calls) == 0  # no physical drive at all


def test_already_at_target_after_remeasure(monkeypatch):
    _patch_measure_and_guard(
        monkeypatch,
        measurements=[(5.0, _LONG_MM), (1.0, 0.1)],
        guards=[None, None],
    )
    _patch_drive_degrees(monkeypatch)
    calls = _patch_change_analysis(monkeypatch, [{"change_description": "drove most of the way"}])

    result = _drive_to()

    assert result["ok"] is True
    assert "no second drive needed" in result["message"]
    assert "drives_executed" not in result
    assert "partial_drive" not in result
    assert len(calls) == 1  # only the first physical drive happened


def test_second_drive_within_touch_offset_skips_second_drive(monkeypatch):
    """Unlike test_already_at_target_after_remeasure's near-zero remeasured
    distance, this uses a moderate but sub-offset remeasured distance to
    exercise the touch-offset clamp specifically (not just a ~0mm reading)."""
    second_distance_mm = _OFFSET_MM * 0.5
    _patch_measure_and_guard(
        monkeypatch,
        measurements=[(5.0, _LONG_MM), (1.0, second_distance_mm)],
        guards=[None, None],
    )
    drive_mock = _patch_drive_degrees(monkeypatch)
    calls = _patch_change_analysis(monkeypatch, [{"change_description": "drove most of the way"}])

    result = _drive_to()

    assert result["ok"] is True
    assert "within the" in result["message"] and "touch offset" in result["message"]
    assert "no second drive needed" in result["message"]
    assert "drives_executed" not in result
    assert len(calls) == 1  # only the first physical drive happened
    drive_mock.assert_called_once()  # first (partial) leg only — no second drive_degrees call
