"""
YOLOE detection of non-COCO targets (grasp_readiness._YOLOE_CLASSES).

The "paper ball" class is a crumpled white paper ball, prompted by a stored
visual-prompt embedding (mcp_robot/yoloe_vpe/paper_ball.pt) built from three
example boxes. Before it existed, target_class_yolo="ball" (COCO sports ball /
orange / apple) never found the ball, so every grasp-readiness gate fell back
to Gemini — ~2s a call, 20 calls/day on the free tier, and on 2026-10-05 a box
~140px off the ball (see tests/test_hybrid_locate.py).

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

# No paper ball in frame, but other white objects (cup, light switches) or
# the robot alone.
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


if __name__ == "__main__":
    unittest.main()
