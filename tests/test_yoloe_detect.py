"""
YOLOE detection of non-COCO targets (grasp_readiness._YOLOE_CLASSES).

The "paper ball" class is a crumpled white paper ball, prompted by a stored
visual-prompt embedding (mcp_robot/yoloe_vpe/paper_ball.pt) built from three
example boxes. Before it existed, target_class_yolo="ball" (COCO sports ball /
orange / apple) never found the ball, so every grasp-readiness gate fell back
to Gemini — ~2s a call, 20 calls/day on the free tier, and on 2026-10-05 a box
~140px off the ball (see tests/test_hybrid_locate.py).

The "colored cup" class is the three waste-bin cups tilted in their rack
(mcp_robot/yoloe_vpe/colored_cup.pt, nine example boxes), and "blue cup" /
"yellow cup" / "green cup" narrow it by the hue of each detection's mask
(grasp_readiness._YOLOE_COLOR_CLASSES). Before them, target_class_yolo="cup"
(COCO cup/bowl/bottle/vase) never found the tilted cups — on 2026-10-11 it
matched the robot's own wheel as a "bottle" (0.50) instead — so every bin
approach fell back to Gemini.

Offline, but needs yoloe-26l-seg.pt (auto-downloaded on first use, ~80 MB).

Run with:
    python -m pytest tests/test_yoloe_detect.py -s
"""
import pathlib
import unittest
from unittest import mock

import cv2

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# Frames NOT among the VPE's examples, with the ball's box measured on a 3×
# zoom of each fixture.
_HELD_OUT = {
    # 2026-10-05 15:16:22 control_gripper gate frame: ball between the jaws.
    "yoloe/paper_ball_in_jaws_heldout.jpg": (503, 738, 577, 797),
    # 2026-10-05 15:14:13: ball on the floor ahead of the gripper.
    "yoloe/paper_ball_ahead_heldout.jpg": (465, 734, 533, 800),
    # Earlier session, DroidCam's oblique view (different camera and angle
    # from every example), ball partly under a gripper finger.
    "grasp_readiness/droidcam_036.jpg": (796, 390, 894, 475),
}

# Frames NOT among the "colored cup" VPE's examples, each cup's box measured
# by colour thresholding (blue H 100-115, yellow 21-32, green 60-95) and
# checked by eye. Three camera placements.
_CUPS_HELD_OUT = {
    # 2026-10-11 15:57:15: robot just below the bins carrying the ball, its
    # yellow plate under a body length from the yellow cup.
    "yoloe/colored_cups_robot_at_bins_heldout.jpg": {
        "blue cup": (82, 82, 211, 272), "yellow cup": (254, 87, 386, 279), "green cup": (434, 94, 563, 284)},
    # 16:02:43: robot holding the ball beside the green cup.
    "yoloe/colored_cups_robot_by_green_cup_heldout.jpg": {
        "blue cup": (82, 82, 211, 272), "yellow cup": (256, 87, 385, 279), "green cup": (434, 94, 563, 284)},
    # 11:4x: ball in the jaws, facing the bins.
    "heading/simpleipcamera_ball_in_jaws_facing_bins.jpg": {
        "blue cup": (99, 177, 221, 353), "yellow cup": (263, 183, 391, 361), "green cup": (436, 192, 559, 367)},
    # 2026-10-05: ribbon cable across the robot's plate.
    "heading/simpleipcamera_ribbon_split_plate.jpg": {
        "blue cup": (110, 53, 242, 248), "yellow cup": (287, 53, 415, 248), "green cup": (465, 54, 599, 250)},
}

# No paper ball in frame, but other white objects (cup, light switches) or
# the robot alone. No bin cups either.
_NO_BALL = (
    "grasp_readiness/droidcam_cup.jpg",
    "grasp_readiness/droidcam_light_switch.png",
    "grasp_readiness/light_switch_cabinet.jpg",
    "navigation/droidcam_nav_obstacle.jpg",
)


def _load(rel: str):
    bgr = cv2.imread(str(FIXTURES / rel))
    assert bgr is not None, f"Could not load {FIXTURES / rel}"
    return bgr


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union else 0.0


class TestYoloePaperBall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mcp_robot import grasp_readiness
        cls.gr = grasp_readiness
        grasp_readiness._load_yoloe_model()

    def test_finds_held_out_balls(self):
        for rel, gt in _HELD_OUT.items():
            with self.subTest(rel):
                objects = self.gr._yolo_detect(_load(rel), target_class="paper ball")
                self.assertTrue(objects, f"{rel}: no paper ball detected")
                best = max(objects, key=lambda o: o.confidence)
                box = (best.x1, best.y1, best.x2, best.y2)
                self.assertGreaterEqual(_iou(box, gt), 0.5,
                                        f"{rel}: best box {box} (conf {best.confidence:.2f}) misses the ball {gt}")

    def test_no_detection_without_ball(self):
        for rel in _NO_BALL:
            with self.subTest(rel):
                objects = self.gr._yolo_detect(_load(rel), target_class="paper ball")
                self.assertEqual(objects, [], f"{rel}: false paper ball(s) {objects}")

    def test_grasp_gate_finds_ball_without_gemini(self):
        """The gate must take the ball from YOLOE: _vlm_detect (Gemini) is the
        fallback for when YOLO finds nothing, and must not be reached."""
        rel = "yoloe/paper_ball_in_jaws_heldout.jpg"
        with mock.patch.object(self.gr, "_vlm_detect", side_effect=AssertionError("Gemini was called")):
            result, _, obj = self.gr._compute_readiness(
                _load(rel), target_class_yolo="paper ball",
                target_class_free_text="small crumpled white paper ball between the gripper fingers",
            )
        self.assertTrue(result.object_detected, result.reason)
        self.assertGreaterEqual(_iou((obj.x1, obj.y1, obj.x2, obj.y2), _HELD_OUT[rel]), 0.5)

    def test_coco_classes_still_use_yolo(self):
        with mock.patch.object(self.gr, "_yoloe_detect", side_effect=AssertionError("YOLOE was called")):
            self.gr._yolo_detect(_load("grasp_readiness/droidcam_cup.jpg"), target_class="cup")

    def test_stored_vpe_matches_examples(self):
        """mcp_robot/yoloe_vpe/paper_ball.pt must still be what its
        _YOLOE_CLASSES examples produce — rebuild it with save_yoloe_vpe()
        after changing the examples or _YOLOE_MODEL_NAME."""
        import torch

        stored = torch.load(self.gr._yoloe_vpe_path("paper ball"), weights_only=True)
        rebuilt = self.gr.build_yoloe_vpe("paper ball")
        cos = torch.nn.functional.cosine_similarity(stored.flatten(), rebuilt.flatten(), dim=0).item()
        self.assertGreater(cos, 0.99, f"stored VPE is stale (cosine {cos:.3f} to a rebuild)")


class TestYoloeColoredCups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mcp_robot import grasp_readiness
        cls.gr = grasp_readiness
        grasp_readiness._load_yoloe_model()

    @staticmethod
    def _box(obj):
        return (obj.x1, obj.y1, obj.x2, obj.y2)

    def test_finds_each_cup_by_colour(self):
        for rel, cups in _CUPS_HELD_OUT.items():
            bgr = _load(rel)
            for colour, gt in cups.items():
                with self.subTest(rel=rel, cup=colour):
                    objects = self.gr._yolo_detect(bgr, target_class=colour)
                    self.assertEqual(len(objects), 1, f"{rel}: {colour}: {objects}")
                    box = self._box(objects[0])
                    self.assertGreaterEqual(_iou(box, gt), 0.85, f"{rel}: {colour} box {box} misses the cup {gt}")

    def test_colored_cup_finds_each_cup_once(self):
        """Any bin: one box per cup — YOLOE's duplicate boxes for one object dropped."""
        for rel, cups in _CUPS_HELD_OUT.items():
            with self.subTest(rel):
                boxes = [self._box(o) for o in self.gr._yolo_detect(_load(rel), target_class="colored cup")]
                self.assertEqual(len(boxes), 3, f"{rel}: {boxes}")
                for colour, gt in cups.items():
                    self.assertTrue(any(_iou(b, gt) >= 0.85 for b in boxes), f"{rel}: no box on the {colour} {gt}: {boxes}")

    def test_orange_cast_frame_keeps_colours(self):
        """2026-10-11 15:27:44, a camera exposure transient: the frame turned orange
        and, read without white balance, the green cup's hue fell in the yellow
        band — a detection on the wrong bin. No colour may land on another cup."""
        cups = {"blue cup": (98, 177, 222, 354), "yellow cup": (264, 182, 388, 361), "green cup": (436, 190, 560, 367)}
        bgr = _load("yoloe/colored_cups_washed_out_heldout.jpg")
        for colour in cups:
            for obj in self.gr._yolo_detect(bgr, target_class=colour):
                for other, gt in cups.items():
                    if other != colour:
                        self.assertLess(_iou(self._box(obj), gt), 0.5, f"'{colour}' detected on the {other}")
        green = self.gr._yolo_detect(bgr, target_class="green cup")
        self.assertTrue(green, "green cup not found")
        self.assertGreaterEqual(_iou(self._box(green[0]), cups["green cup"]), 0.85)

    def test_no_colored_cups_without_bins(self):
        for rel in _NO_BALL:
            with self.subTest(rel):
                self.assertEqual(self.gr._yolo_detect(_load(rel), target_class="colored cup"), [])

    def test_cup_angle_without_gemini(self):
        """get_robot_state / turn_to / drive_to measure the target through
        annotate_frame_with_object: the green cup must come from YOLOE, and the
        robot's heading must be found beside the yellow cup (see
        test_heading_annotate.TestYellowCupErasedFromBody)."""
        with mock.patch.object(self.gr, "_vlm_detect", side_effect=AssertionError("Gemini was called")):
            _, angle, _, dist_px, _, _ = self.gr.annotate_frame_with_object(
                _load("yoloe/colored_cups_robot_by_green_cup_heldout.jpg"),
                target_class_yolo="green cup",
                target_class_free_text="green plastic cup, rightmost of three in the rack",
            )
        self.assertIsNotNone(angle, "no angle — heading or the green cup not found")
        # Gripper points NNE (~67° from east); the cup's center (497, 189) is
        # ~99° from the body center (553, 526): ~32° CCW.
        self.assertTrue(-40 <= angle <= -25, f"angle={angle:.1f}")
        self.assertTrue(300 <= dist_px <= 380, f"dist_px={dist_px:.0f}")

    def test_stored_vpe_matches_examples(self):
        """mcp_robot/yoloe_vpe/colored_cup.pt must still be what its
        _YOLOE_CLASSES examples produce — rebuild it with
        save_yoloe_vpe("colored cup") after changing them."""
        import torch

        stored = torch.load(self.gr._yoloe_vpe_path("colored cup"), weights_only=True)
        rebuilt = self.gr.build_yoloe_vpe("colored cup")
        cos = torch.nn.functional.cosine_similarity(stored.flatten(), rebuilt.flatten(), dim=0).item()
        self.assertGreater(cos, 0.99, f"stored VPE is stale (cosine {cos:.3f} to a rebuild)")


if __name__ == "__main__":
    unittest.main()
