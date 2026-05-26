"""
Visualise detect_heading results on droidcam fixture images.

Saves annotated copies (green forward arrow) to tests/fixtures/heading/annotated/
for manual inspection.  Run with:

    python -m pytest tests/test_heading_annotate.py -s
"""
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


if __name__ == "__main__":
    unittest.main()
