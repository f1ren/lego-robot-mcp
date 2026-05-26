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

FIXTURES      = pathlib.Path(__file__).parent / "fixtures"
FAR_BALL      = FIXTURES / "grasp_readiness" / "grasp_not_ready_far.jpg"
NEARBY_BALL   = FIXTURES / "grasp_readiness" / "static_video" / "droidcam_000.jpg"
CUP_IMG       = FIXTURES / "grasp_readiness" / "droidcam_cup.jpg"
ANNOTATED_DIR = FIXTURES / "grasp_readiness" / "annotated"


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


class TestGraspReadinessCup(unittest.TestCase):
    """Run check_grasp_readiness on the live-captured cup fixture."""

    @classmethod
    def setUpClass(cls):
        from mcp_robot.grasp_readiness import _compute_readiness, _load_model
        _load_model()
        ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
        bgr = _load(CUP_IMG)
        cls._bgr = bgr
        cls._result, cls._heading, cls._obj = _compute_readiness(bgr, target_class="cup")
        print(f"\n[cup fixture] {cls._result.to_text()}")
        cls._save_annotated(bgr, cls._result, cls._heading, cls._obj)

    @staticmethod
    def _save_annotated(bgr, result, heading, obj):
        import math
        from mcp_robot.heading import annotate_bgr as _heading_annotate_bgr
        # Use the exact same arrow-drawing code as the main flow.
        out = _heading_annotate_bgr(bgr.copy()) if heading is not None else bgr.copy()

        if heading is not None:
            bx, by = heading.body_center
            fw = heading.forward
            # label the tip of the forward arrow
            h_img, w_img = out.shape[:2]
            diag = (w_img**2 + h_img**2) ** 0.5
            fwd_len = int(diag * 0.45)
            fwd_tip = (
                max(0, min(w_img - 1, int(bx + fw[0] * fwd_len))),
                max(0, min(h_img - 1, int(by + fw[1] * fwd_len))),
            )
            cv2.putText(out, "fw", (fwd_tip[0] + 4, fwd_tip[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)

            if obj is not None:
                ox, oy = obj.center
                # blue arrow: body centre → object centre
                cv2.arrowedLine(out, (bx, by), (ox, oy), (220, 100, 0), 3, tipLength=0.12)
                cv2.putText(out, "obj", (ox + 6, oy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 100, 0), 2, cv2.LINE_AA)

                # object bbox
                cv2.rectangle(out, (obj.x1, obj.y1), (obj.x2, obj.y2), (0, 200, 255), 2)
                label = f"{obj.class_name} {obj.confidence:.0%}"
                cv2.putText(out, label, (obj.x1, obj.y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

                # angle arc annotation between fw and to-object vectors
                dx, dy = ox - bx, oy - by
                t = dx * fw[0] + dy * fw[1]
                perp_x = dx - t * fw[0]
                perp_y = dy - t * fw[1]
                perp_dist = math.hypot(perp_x, perp_y)
                angle_deg = math.degrees(math.atan2(perp_dist, max(t, 1.0)))
                cross = fw[0] * dy - fw[1] * dx
                rot_dir = "CW" if cross > 0 else "CCW"
                angle_txt = f"rot: {angle_deg:.0f}deg {rot_dir}"
                cv2.putText(out, angle_txt, (bx + 8, by - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        verdict = "READY" if result.ready else "NOT READY"
        color = (0, 200, 0) if result.ready else (0, 0, 220)
        cv2.putText(out, verdict, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        out_path = ANNOTATED_DIR / "droidcam_cup_grasp_readiness.jpg"
        cv2.imwrite(str(out_path), out)
        print(f"  -> saved {out_path}")

    def test_cup_is_detected(self):
        """object_detected must be True — cup is clearly visible in the fixture."""
        self.assertTrue(self._result.object_detected,
                        f"Cup not detected. reason: {self._result.reason}")

    def test_cup_class_is_synonym(self):
        """Detected class must be a cup synonym (cup / bottle / vase)."""
        self.assertIn(self._result.object_class, {"cup", "bottle", "vase"},
                      f"Unexpected class: {self._result.object_class!r}")

    def test_cup_confidence_above_threshold(self):
        self.assertGreater(self._result.object_confidence, 0.20,
                           f"Confidence too low: {self._result.object_confidence:.2f}")

    def test_not_ready_cup_not_in_grasp_position(self):
        """Cup is on the right, robot on the left — should not be ready."""
        self.assertFalse(self._result.ready,
                         f"Expected NOT ready, got: {self._result.reason}")

    def test_actionable_feedback_provided(self):
        """When not ready, action must be non-empty."""
        if not self._result.ready:
            self.assertTrue(self._result.action,
                            "Expected non-empty action for not-ready result")

    def test_to_dict_schema(self):
        d = self._result.to_dict()
        for key in ("ready", "reason", "action", "object_detected", "checks", "metrics"):
            self.assertIn(key, d, f"Missing key '{key}' in to_dict()")


if __name__ == "__main__":
    unittest.main()
