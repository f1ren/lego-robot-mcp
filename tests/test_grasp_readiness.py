"""
Unit tests for grasp_readiness.check_grasp_readiness.

Both fixture images must return ready=False:
  - grasp_not_ready_far.jpg   : ball is far from the robot (bottom-right corner)
  - static_video/droidcam_000.jpg : ball is visible but not touching the robot body

These tests exercise the full CV pipeline (YOLO + heading detection).
YOLO weights are downloaded automatically on first run (~6 MB, yolo11n.pt).
"""
import math
import pathlib
import unittest
from unittest import mock

import cv2

FIXTURES      = pathlib.Path(__file__).parent / "fixtures"
FAR_BALL      = FIXTURES / "grasp_readiness" / "grasp_not_ready_far.jpg"
NEARBY_BALL   = FIXTURES / "static_video" / "droidcam_000.jpg"
CUP_IMG       = FIXTURES / "grasp_readiness" / "droidcam_cup.jpg"
CUP_TOO_FAR_IMG = FIXTURES / "grasp_readiness" / "droidcam_cup_too_far.jpg"
CUP_TOUCHING_IMG = FIXTURES / "grasp_readiness" / "droidcam_cup_touching.jpg"
CUP_PARTIAL_OCCLUSION_IMG = FIXTURES / "grasp_readiness" / "droidcam_cup_partial_occlusion.jpg"
ANNOTATED_DIR = FIXTURES / "grasp_readiness" / "annotated"


def _load(path: pathlib.Path):
    bgr = cv2.imread(str(path))
    assert bgr is not None, f"Could not load fixture: {path}"
    return bgr


def _save_annotated(bgr, result, heading, obj, out_path: pathlib.Path):
    """Draw arrow + object bbox + angle/distance readout and save for manual inspection.

    Shared by any grasp-readiness test that wants a visual record of what
    _compute_readiness saw — same arrow-drawing code as the main flow.
    """
    from mcp_robot.heading import annotate_bgr as _heading_annotate_bgr
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
    if result.missing_distance_mm:
        cv2.putText(out, f"~{result.missing_distance_mm:.0f}mm to go", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"  -> saved {out_path}")


class TestGraspReadiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import here so YOLO model is loaded once for all tests.
        from mcp_robot.grasp_readiness import check_grasp_readiness, _load_model
        cls._check = staticmethod(check_grasp_readiness)
        _load_model()  # pre-download weights if needed

    def check(self, bgr):
        return self._check(bgr, target_class_yolo="ball")

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
        cls._result, cls._heading, cls._obj = _compute_readiness(bgr, target_class_yolo="cup")
        print(f"\n[cup fixture] {cls._result.to_text()}")
        _save_annotated(bgr, cls._result, cls._heading, cls._obj,
                         ANNOTATED_DIR / "droidcam_cup_grasp_readiness.jpg")

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


class TestGraspReadinessCupTooFarRegression(unittest.TestCase):
    """
    Regression for a real false positive (output/logs/mcp_server.log,
    2026-07-09 07:39:22,042): YOLO found no cup, the Gemini/CV-refine VLM
    fallback located one whose nearest point measured ~112mm from the
    robot's front, and the OLD pixel-diagonal-fraction touch heuristic
    (112px against a ~117px image-diagonal threshold) reported ready=True —
    even though the cup was visibly not touching the robot.

    _vlm_detect is stubbed with that exact logged detection (same bbox,
    centroid, confidence, note) so this test is deterministic/offline and
    exercises only _compute_readiness's mm-calibrated distance logic — the
    part that was actually buggy — without a live Gemini call.
    """

    @classmethod
    def setUpClass(cls):
        from mcp_robot import grasp_readiness as grasp_mod
        from mcp_robot.grasp_readiness import DetectedObject
        grasp_mod._load_model()

        bgr = _load(CUP_TOO_FAR_IMG)
        cls._bgr = bgr

        logged_detection = DetectedObject(
            class_name="small blue plastic cup",
            confidence=0.95,
            x1=358, y1=328, x2=431, y2=427,
            note=(
                "The small blue plastic cup is located towards the upper "
                "right of the image, sitting on the wooden floor near the wall."
            ),
            contact_px=(394, 372),
        )
        with mock.patch.object(grasp_mod, "_vlm_detect", return_value=logged_detection):
            cls._result, cls._heading, cls._obj = grasp_mod._compute_readiness(
                bgr, target_class_yolo="cup", target_class_free_text="small blue plastic cup",
            )
        print(f"\n[cup-too-far regression] {cls._result.to_text()}")
        _save_annotated(bgr, cls._result, cls._heading, cls._obj,
                         ANNOTATED_DIR / "droidcam_cup_too_far_grasp_readiness.jpg")

    def test_not_ready_cup_too_far(self):
        """Cup is ~112mm from the robot's front — must NOT be ready.

        This is the exact case that used to report ready=True.
        """
        self.assertFalse(
            self._result.ready,
            f"Expected NOT ready (cup too far to grasp), got: {self._result.reason}",
        )

    def test_missing_distance_reported(self):
        """Not-ready-by-distance must report how far to drive, in mm."""
        from mcp_robot import config
        self.assertIsNotNone(self._result.missing_distance_mm)
        self.assertGreater(self._result.missing_distance_mm, 0)
        # Sanity bound: the measured gap is nowhere near a full body length off.
        self.assertLess(self._result.missing_distance_mm, config.ROBOT_BODY_LENGTH_MM)

    def test_action_mentions_distance(self):
        """Actionable feedback must state the drive distance, not just 'drive forward'."""
        self.assertIn("mm", self._result.action)

    def test_to_dict_includes_missing_distance_mm(self):
        d = self._result.to_dict()
        self.assertIn("missing_distance_mm", d["metrics"])
        self.assertIsNotNone(d["metrics"]["missing_distance_mm"])


class TestGraspReadinessCupTouchingRegression(unittest.TestCase):
    """
    Regression for a real false negative (output/logs/mcp_server.log,
    2026-07-09 16:51:35,477): the blue cup sits directly against the
    robot's front, right next to the gripper — visibly in grasp position —
    yet the mm-calibrated touch-distance check reported ready=False
    (front-gap=51mm, ~11mm past the GRASP_TOUCH_THRESHOLD_MM cutoff that
    was 40mm at the time). This case is what raised the threshold to 60mm
    — see config.GRASP_TOUCH_THRESHOLD_MM.

    _vlm_detect is stubbed with that exact logged detection (same bbox,
    centroid, confidence, note — from locate_object_hybrid's CV-refined
    result, not a live Gemini call) so this test is deterministic/offline
    and exercises only _compute_readiness's distance/alignment logic.
    """

    @classmethod
    def setUpClass(cls):
        from mcp_robot import grasp_readiness as grasp_mod
        from mcp_robot.grasp_readiness import DetectedObject
        grasp_mod._load_model()

        bgr = _load(CUP_TOUCHING_IMG)
        cls._bgr = bgr

        logged_detection = DetectedObject(
            class_name="blue cup",
            confidence=0.95,
            x1=408, y1=249, x2=480, y2=326,
            note=(
                "A blue cup is visible on the floor next to a piece of "
                "furniture and a small robot-like device."
            ),
            contact_px=(441, 283),
            outer_bbox=(396, 254, 471, 413),
        )
        with mock.patch.object(grasp_mod, "_vlm_detect", return_value=logged_detection):
            cls._result, cls._heading, cls._obj = grasp_mod._compute_readiness(
                bgr, target_class_yolo="cup", target_class_free_text="blue cup",
            )
        print(f"\n[cup-touching regression] {cls._result.to_text()}")
        _save_annotated(bgr, cls._result, cls._heading, cls._obj,
                         ANNOTATED_DIR / "droidcam_cup_touching_grasp_readiness.jpg")

    def test_ready_cup_touching_body(self):
        """Cup is resting against the robot's front — must be ready.

        This is the exact case that used to report ready=False.
        """
        self.assertTrue(
            self._result.ready,
            f"Expected READY (cup touching robot body), got: {self._result.reason}",
        )

    def test_to_dict_schema(self):
        d = self._result.to_dict()
        for key in ("ready", "reason", "action", "object_detected", "checks", "metrics"):
            self.assertIn(key, d, f"Missing key '{key}' in to_dict()")


class TestGraspReadinessCupPartialOcclusionRegression(unittest.TestCase):
    """
    Regression for a real gap-exaggeration bug (output/logs/mcp_server.log,
    2026-07-09 17:21:46,991): the blue cup sits right next to the robot's
    gripper, but the gripper's own claws occlude the cup's lower half from the
    external camera. Gemini's rough VLM box (356,253,398,322) already only
    covered the visible upper rim, and cv_refine_location's HSV pass on top of
    it shrank the box further to (373,258,415,311) — under-segmenting even the
    visible part. That undersized box fed touches_body/arrow_well_over
    directly, inflating the reported front-gap to 83mm and the alignment
    offset past threshold — both checks failed by more than the true (still
    imperfect, but much smaller) gap warranted.

    _vlm_detect is stubbed with that exact logged DetectedObject — same bbox,
    contact_px, outer_bbox all taken from locate_object_hybrid's real output
    for this frame — so this test is deterministic/offline and exercises only
    _compute_readiness's outer_bbox-union fix, without a live Gemini call.
    """

    @classmethod
    def setUpClass(cls):
        from mcp_robot import grasp_readiness as grasp_mod
        from mcp_robot.grasp_readiness import DetectedObject
        grasp_mod._load_model()

        cls._grasp_mod = grasp_mod
        cls._bgr = _load(CUP_PARTIAL_OCCLUSION_IMG)

        cls._with_union = DetectedObject(
            class_name="blue cup",
            confidence=0.95,
            x1=373, y1=258, x2=415, y2=311,
            note=(
                "The blue cup is visible in the upper center-left portion of "
                "the image, placed on the wooden floor near the white wall."
            ),
            contact_px=(392, 282),
            outer_bbox=(356, 253, 398, 322),
        )
        # Same detection, but as it would look before the outer_bbox union fix
        # (no coarser box to fall back on) — isolates the fix's effect on this
        # exact real frame instead of comparing against a hardcoded number.
        cls._without_union = DetectedObject(
            class_name="blue cup",
            confidence=0.95,
            x1=373, y1=258, x2=415, y2=311,
            note=cls._with_union.note,
            contact_px=(392, 282),
            outer_bbox=None,
        )

        with mock.patch.object(grasp_mod, "_vlm_detect", return_value=cls._with_union):
            cls._result, cls._heading, cls._obj = grasp_mod._compute_readiness(
                cls._bgr, target_class_yolo="cup", target_class_free_text="blue cup",
            )
        print(f"\n[cup-partial-occlusion regression] {cls._result.to_text()}")
        _save_annotated(cls._bgr, cls._result, cls._heading, cls._obj,
                         ANNOTATED_DIR / "droidcam_cup_partial_occlusion_grasp_readiness.jpg")

    def test_front_gap_shrinks_relative_to_undersized_bbox(self):
        """Without the union, this exact detection reproduces the logged
        incident's front-gap (dist_to_front_px=73, ready=False). With the
        union, the measured gap to the robot's front must shrink — the object
        didn't move, only the box's known extent did."""
        with mock.patch.object(self._grasp_mod, "_vlm_detect", return_value=self._without_union):
            baseline, _, _ = self._grasp_mod._compute_readiness(
                self._bgr, target_class_yolo="cup", target_class_free_text="blue cup",
            )
        self.assertEqual(baseline.dist_to_front_px, 73.0,
                          "baseline (no union) should reproduce the exact logged incident")
        self.assertLess(
            self._result.dist_to_front_px, baseline.dist_to_front_px,
            "outer_bbox union should shrink the measured front-gap, not grow it",
        )

    def test_missing_distance_much_smaller_than_logged_incident(self):
        """Logged incident originally reported ~23mm missing to close the gap
        (front-gap 83mm, threshold 60mm). The union must bring that down
        noticeably, even though it doesn't need to close it entirely."""
        self.assertIsNotNone(self._result.missing_distance_mm)
        self.assertLess(
            self._result.missing_distance_mm, 15.0,
            f"Expected missing_distance_mm well below the originally-logged "
            f"~23mm, got {self._result.missing_distance_mm:.1f}mm",
        )

    def test_still_not_ready_this_frame_needs_more_driving(self):
        """The union corrects the *magnitude* of the gap — it doesn't fabricate
        contact that isn't there. This exact frame is genuinely still a few mm
        short of touching, so ready must stay False here."""
        self.assertFalse(self._result.ready, f"got: {self._result.reason}")

    def test_action_still_mentions_distance(self):
        self.assertIn("mm", self._result.action)

    def test_to_dict_schema(self):
        d = self._result.to_dict()
        for key in ("ready", "reason", "action", "object_detected", "checks", "metrics"):
            self.assertIn(key, d, f"Missing key '{key}' in to_dict()")


if __name__ == "__main__":
    unittest.main()
