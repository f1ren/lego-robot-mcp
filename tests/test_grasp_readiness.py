"""
Unit tests for grasp_readiness.check_grasp_readiness.

Both fixture images must return ready=False:
  - grasp_not_ready_far.jpg   : ball is far from the robot (bottom-right corner)
  - static_video/droidcam_000.jpg : ball is visible but not touching the robot body

These tests exercise the full CV pipeline (YOLO + heading detection).
YOLO weights are downloaded automatically on first run (~6 MB, yolo11n.pt).
"""
import pathlib
import unittest

import cv2

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
FAR_BALL = FIXTURES / "grasp_not_ready_far.jpg"
NEARBY_BALL = FIXTURES / "static_video" / "droidcam_000.jpg"


def _load(path: pathlib.Path):
    bgr = cv2.imread(str(path))
    assert bgr is not None, f"Could not load fixture: {path}"
    return bgr


class TestGraspReadiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import here so YOLO model is loaded once for all tests.
        from mcp_robot.grasp_readiness import check_grasp_readiness, _load_model
        cls._check = staticmethod(check_grasp_readiness)
        _load_model()  # pre-download weights if needed

    def check(self, bgr):
        return self._check(bgr)

    def test_not_ready_ball_far_from_robot(self):
        """Ball is in the bottom-right corner, robot is in centre — not ready."""
        result = self.check(_load(FAR_BALL))
        self.assertFalse(
            result.ready,
            f"Expected NOT ready (ball far from robot), got: {result.reason}",
        )

    def test_not_ready_ball_not_touching_body(self):
        """Ball is visible near the gripper but has a clear gap — not ready."""
        result = self.check(_load(NEARBY_BALL))
        self.assertFalse(
            result.ready,
            f"Expected NOT ready (ball not touching robot body), got: {result.reason}",
        )

    def test_returns_actionable_text_when_not_ready(self):
        """Not-ready result must include a non-empty action field."""
        for path in (FAR_BALL, NEARBY_BALL):
            with self.subTest(fixture=path.name):
                result = self.check(_load(path))
                if not result.ready:
                    self.assertTrue(
                        result.action,
                        f"Expected non-empty action for not-ready result on {path.name}",
                    )

    def test_to_text_contains_verdict(self):
        """to_text() output must contain the readiness verdict."""
        result = self.check(_load(FAR_BALL))
        text = result.to_text()
        self.assertIn("NOT READY", text)

    def test_to_dict_schema(self):
        """to_dict() must have the expected top-level keys."""
        result = self.check(_load(FAR_BALL))
        d = result.to_dict()
        for key in ("ready", "reason", "action", "object_detected", "checks", "metrics"):
            self.assertIn(key, d, f"Missing key '{key}' in to_dict() output")


if __name__ == "__main__":
    unittest.main()
