"""
Ground-truth unit tests for heading.compute_heading_to_object_angle.

This is the pure-geometry function behind the "Heading analysis: ... object
at X.Y deg CW/CCW from forward" log line emitted by capture_simpleipcamera_still
(mcp_robot/camera.py) — the number `click_button` and friends use to decide
how far to turn. It has no camera/YOLO dependency, so it's pinned here with
synthetic Heading + object-center inputs instead of image fixtures, which
lets every case assert an exact expected angle rather than "looks about
right".

Sign convention under test (from the function's own docstring):
    Positive -> object is clockwise from forward (image y-down, viewed from above)
    Negative -> counter-clockwise

Expected angles are derived independently via a clockwise rotation matrix in
image (y-down) coordinates:
    rotate_cw(f, deg)_x = fx*cos(deg) - fy*sin(deg)
    rotate_cw(f, deg)_y = fx*sin(deg) + fy*cos(deg)
Algebraically, for unit f, compute_heading_to_object_angle(heading, body_center
+ k*rotate_cw(f, deg)) == deg for any k > 0:
    dot   = f . rotate_cw(f, deg) = cos(deg)   [since |f| = 1]
    cross = f x rotate_cw(f, deg) = sin(deg)
    atan2(cross, dot) = atan2(sin(deg), cos(deg)) = deg
So a mismatch here means the implementation's sign/argument-order has
drifted from its documented contract, not that the test's expectations
were reverse-engineered from current behaviour.
"""
import math
import unittest

from mcp_robot.heading import Heading, compute_heading_to_object_angle


def _rotate_cw(vec: tuple[float, float], deg: float) -> tuple[float, float]:
    """Rotate `vec` by `deg` clockwise in image (y-down) coordinates."""
    a = math.radians(deg)
    fx, fy = vec
    cos_a, sin_a = math.cos(a), math.sin(a)
    return (fx * cos_a - fy * sin_a, fx * sin_a + fy * cos_a)


def _make_heading(forward: tuple[float, float], body_center=(0, 0)) -> Heading:
    return Heading(
        body_center=body_center,
        arrow_anchor=body_center,
        forward=forward,
        body_area=1000,
        gripper_area=100,
    )


class TestComputeHeadingToObjectAngleCardinals(unittest.TestCase):
    """Named cases for the intuitive directions — easy to eyeball/verify by hand."""

    def _assert_angle(self, forward, deg, distance=100.0, places=3):
        heading = _make_heading(forward)
        bx, by = heading.body_center
        ox, oy = _rotate_cw(forward, deg)
        obj_center = (bx + ox * distance, by + oy * distance)
        got = compute_heading_to_object_angle(heading, obj_center)
        if abs(abs(deg) - 180) < 1e-9:
            # atan2 returns the +-180 boundary as either sign — both correct.
            self.assertAlmostEqual(abs(got), 180.0, places=places)
        else:
            self.assertAlmostEqual(
                got, deg, places=places,
                msg=f"forward={forward} expected={deg} got={got}",
            )

    def test_object_dead_ahead_is_zero(self):
        self._assert_angle((0.0, -1.0), 0.0)

    def test_object_dead_behind_is_180(self):
        self._assert_angle((0.0, -1.0), 180.0)

    def test_object_90_clockwise(self):
        """East relative to a north-pointing forward -- CW like N->E on a compass."""
        self._assert_angle((0.0, -1.0), 90.0)

    def test_object_90_counterclockwise(self):
        """West relative to a north-pointing forward -- CCW like N->W on a compass."""
        self._assert_angle((0.0, -1.0), -90.0)

    def test_object_45_clockwise(self):
        self._assert_angle((0.0, -1.0), 45.0)

    def test_object_45_counterclockwise(self):
        self._assert_angle((0.0, -1.0), -45.0)


class TestComputeHeadingToObjectAngleSweep(unittest.TestCase):
    """Broad sweep across forward vectors and angles, via the rotation-matrix proof above."""

    FORWARDS = [
        (0.0, -1.0),                                   # north
        (1.0, 0.0),                                     # east
        (math.sqrt(0.5), math.sqrt(0.5)),                # south-east
        (-0.6, 0.8),                                     # arbitrary unit vector
    ]
    ANGLES = list(range(-165, 180, 15))  # -165 .. 165 step 15, avoids the +-180 boundary

    def test_sweep(self):
        for forward in self.FORWARDS:
            heading = _make_heading(forward)
            for deg in self.ANGLES:
                with self.subTest(forward=forward, deg=deg):
                    ox, oy = _rotate_cw(forward, deg)
                    obj_center = (ox * 250.0, oy * 250.0)
                    got = compute_heading_to_object_angle(heading, obj_center)
                    self.assertAlmostEqual(got, float(deg), places=3)


class TestComputeHeadingToObjectAngleInvariants(unittest.TestCase):
    """Properties the angle must hold regardless of where things sit in the image."""

    def test_distance_does_not_affect_angle(self):
        """A near object and a far object on the same ray report the same angle."""
        heading = _make_heading((0.0, -1.0))
        near = compute_heading_to_object_angle(heading, (30.0, -60.0))
        far = compute_heading_to_object_angle(heading, (300.0, -600.0))
        self.assertAlmostEqual(near, far, places=6)

    def test_angle_is_relative_to_body_center_not_absolute_position(self):
        """Shifting body_center and the object by the same offset must not change the angle."""
        forward = (0.0, -1.0)
        heading_origin = _make_heading(forward, body_center=(0, 0))
        heading_shifted = _make_heading(forward, body_center=(500, 400))
        angle_origin = compute_heading_to_object_angle(heading_origin, (50, -50))
        angle_shifted = compute_heading_to_object_angle(heading_shifted, (550, 350))
        self.assertAlmostEqual(angle_origin, angle_shifted, places=6)


if __name__ == "__main__":
    unittest.main()
