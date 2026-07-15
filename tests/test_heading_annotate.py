"""
Visualise detect_heading results on droidcam fixture images.

Saves annotated copies (green forward arrow) to tests/fixtures/heading/annotated/
for manual inspection.  Run with:

    python -m pytest tests/test_heading_annotate.py -s
"""
import math
import pathlib
import unittest

import cv2

FIXTURES  = pathlib.Path(__file__).parent / "fixtures"
OUT_DIR   = FIXTURES / "heading" / "annotated"

# One representative image per scenario, plus the cup fixture that currently fails.
IMAGES = sorted({
    p for pattern in (
        "grasp_readiness/droidcam*.jpg",
        "static_video/droidcam_000.jpg",
        "arm_motion/droidcam_004.jpg",
        "drive_motion/droidcam_005.jpg",
        "gripper_motion/droidcam_008.jpg",
        "move_forward/droidcam_000.jpg",
        "navigation/droidcam_robot_hiding_switch.jpg",
    )
    for p in FIXTURES.glob(pattern)
    if "annotated" not in p.parts
})


class TestHeadingAnnotate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import annotate_bgr, detect_heading
        cls._annotate = staticmethod(annotate_bgr)
        cls._detect   = staticmethod(detect_heading)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_annotate_all_fixtures(self):
        self.assertTrue(IMAGES, f"No fixture images found")
        for img_path in IMAGES:
            with self.subTest(image=img_path.name):
                bgr = cv2.imread(str(img_path))
                self.assertIsNotNone(bgr, f"Could not load {img_path}")

                heading = self._detect(bgr)
                status = (
                    f"body_center={heading.body_center} "
                    f"forward=({heading.forward[0]:.2f},{heading.forward[1]:.2f})"
                    if heading else "NO HEADING DETECTED"
                )
                print(f"\n{img_path.parent.name}/{img_path.name}: {status}")

                annotated = self._annotate(bgr)
                # stamp the subfolder name so images from different dirs don't collide
                stem = f"{img_path.parent.name}__{img_path.stem}"
                out_path = OUT_DIR / f"{stem}.jpg"
                cv2.imwrite(str(out_path), annotated)
                print(f"  -> saved {out_path}")


class TestHeadingDirection(unittest.TestCase):
    """Assert that detect_heading returns the correct compass direction for known fixtures."""

    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import detect_heading
        cls._detect = staticmethod(detect_heading)

    def _compass(self, forward: tuple[float, float]) -> str:
        dx, dy = forward
        angle = math.degrees(math.atan2(-dy, dx))  # +x=E, -y=N (image y-down)
        labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        return labels[round(angle / 45) % 8]

    def _assert_heading(self, rel_path: str, expected_compass: str):
        img_path = FIXTURES / rel_path
        bgr = cv2.imread(str(img_path))
        self.assertIsNotNone(bgr, f"Could not load {img_path}")
        heading = self._detect(bgr)
        self.assertIsNotNone(heading, f"No heading detected in {rel_path}")
        got = self._compass(heading.forward)
        self.assertEqual(
            got, expected_compass,
            f"{rel_path}: expected {expected_compass}, got {got} "
            f"(forward={heading.forward[0]:.3f},{heading.forward[1]:.3f})",
        )

    def test_gripper_west_current(self):
        """Gripper points west — regression for off-axis cable blob bug."""
        self._assert_heading("heading/droidcam_current.jpg", "W")

    def test_gripper_east_drive_motion(self):
        self._assert_heading("drive_motion/droidcam_005.jpg", "E")

    def test_gripper_east_move_forward(self):
        self._assert_heading("move_forward/droidcam_000.jpg", "E")

    def test_gripper_east_static(self):
        self._assert_heading("static_video/droidcam_000.jpg", "E")

    def test_gripper_north_robot_hiding_switch(self):
        """Gripper points north — robot hiding the switch, new navigation fixture."""
        self._assert_heading("navigation/droidcam_robot_hiding_switch.jpg", "N")

    def test_gripper_north_heading_switch_and_lights_on(self):
        """Gripper points north — fixture captured 2026-06-04 with switch and lights on."""
        self._assert_heading("heading/droidcam_heading_switch_and_lights_on.jpg", "N")

    def test_gripper_southwest_cup_shadow(self):
        """Gripper points south-west, toward the cup — regression for the robot's own cast
        shadow (bigger + smoother than the real gripper blob) winning the area-based score
        and flipping the arrow 180 degrees to point NE at the shadow instead."""
        self._assert_heading("heading/droidcam_cup_gripper_shadow.jpg", "SW")


if __name__ == "__main__":
    unittest.main()
