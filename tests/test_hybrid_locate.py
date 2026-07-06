"""
Visualise locate_object_hybrid results on fixture images.

Saves an annotated copy to tests/fixtures/grasp_readiness/annotated/ for
manual inspection.  Run with:

    python -m pytest tests/test_hybrid_locate.py -s
"""
import pathlib
import unittest

import cv2
import numpy as np
import pytest

FIXTURES  = pathlib.Path(__file__).parent / "fixtures" / "grasp_readiness"
OUT_DIR   = FIXTURES / "annotated"
SNAP_CUP  = pathlib.Path("output/snapshots/navigate_to_20260616_095547/step_00_a_raw.jpg")

# Manually verified switch location in droidcam_light_switch.png (confirmed by
# visual inspection of the fixture, generous margin around the switch itself).
# Used as ground truth instead of the VLM's own rough bbox: that bbox has real
# call-to-call jitter, so checking a result against it would only test
# VLM/CV self-consistency, not whether the switch was actually found.
_SWITCH_GROUND_TRUTH_BBOX = (370, 260, 480, 360)


def _annotate(bgr, rough_bbox, refined_bbox, centroid):
    out = bgr.copy()
    rx1, ry1, rx2, ry2 = rough_bbox
    cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 220, 220), 2)
    cv2.putText(out, "VLM rough", (rx1, max(ry1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)

    if refined_bbox is not None:
        bx1, by1, bx2, by2 = refined_bbox
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 200, 0), 2)
        cv2.putText(out, "CV refined", (bx1, max(by1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

    if centroid is not None:
        cx, cy = centroid
        cv2.drawMarker(out, (cx, cy), (0, 200, 0), cv2.MARKER_CROSS, 28, 3)
        cv2.circle(out, (cx, cy), 14, (0, 200, 0), 2)

    return out


@pytest.mark.live
class TestHybridLocate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mcp_robot import vision
        cls.vision = vision
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _run_one(self, img_path, description):
        if not img_path.exists():
            self.skipTest(f"Fixture not found: {img_path}")

        bgr = cv2.imread(str(img_path))
        self.assertIsNotNone(bgr, f"Could not load {img_path}")

        try:
            vlm = self.vision.locate_object_vlm(bgr, description)
        except self.vision.LowConfidenceDetection as exc:
            self.skipTest(f"{img_path.name}: {exc}")
        self.assertIsNotNone(vlm, f"VLM returned None for {img_path.name}")
        rough_bbox, conf, note, hsv_lo, hsv_hi, area_frac = vlm
        print(f"\n{img_path.name}:")
        print(f"  VLM rough bbox : {rough_bbox}  conf={conf:.2f}")
        print(f"  HSV lo={hsv_lo}  hi={hsv_hi}  area_frac={area_frac:.4f}")
        print(f"  note: {note}")

        refined = self.vision.cv_refine_location(bgr, rough_bbox, hsv_lo, hsv_hi, area_frac)
        if refined is not None:
            refined_bbox, centroid = refined
            print(f"  CV refined bbox: {refined_bbox}  centroid={centroid}")
        else:
            refined_bbox, centroid = None, None
            print("  CV refinement: no blob found")

        out = _annotate(bgr, rough_bbox, refined_bbox, centroid)
        out_path = OUT_DIR / f"{img_path.stem}_hybrid.jpg"
        cv2.imwrite(str(out_path), out)
        print(f"  -> saved {out_path}")
        return centroid, rough_bbox

    def test_blue_cup_snapshot(self):
        """Primary assertion: blue cup clearly visible in the nav snapshot."""
        centroid, _ = self._run_one(SNAP_CUP, "blue cup on the floor")
        self.assertIsNotNone(centroid, "CV refinement should find the blue cup in the snapshot")

    def test_white_cup_fixture(self):
        """droidcam_cup.jpg has a white cup — tests VLM finds it and CV refines."""
        self._run_one(FIXTURES / "droidcam_cup.jpg", "white cup on the floor")

    def test_paper_ball_fixture(self):
        """droidcam_036.jpg has a crumpled paper ball — tests a non-cup target."""
        self._run_one(FIXTURES / "droidcam_036.jpg", "crumpled white paper ball on the floor")

    def test_droidcam_000(self):
        """droidcam_000.jpg — blue cup may or may not be present."""
        self._run_one(FIXTURES / "droidcam_000.jpg", "blue cup on the floor")

    def test_white_light_switch_fixture(self):
        """droidcam_light_switch.png has a white wall light switch — non-COCO
        target, same free-text description click_button uses (see server.py's
        target_class_free_text for click_button). The switch sits right where
        sunlight hits the wall, so a same-brightness patch of floor/wall right
        next to it is a trap for HSV-based CV refinement."""
        centroid, _ = self._run_one(
            FIXTURES / "droidcam_light_switch.png", "white light switch on the wall")
        self.assertIsNotNone(centroid, "CV refinement should find the white light switch in the fixture")
        gx1, gy1, gx2, gy2 = _SWITCH_GROUND_TRUTH_BBOX
        cx, cy = centroid
        self.assertTrue(
            gx1 <= cx <= gx2 and gy1 <= cy <= gy2,
            f"Refined centroid {centroid} should land on the switch (ground truth "
            f"{_SWITCH_GROUND_TRUTH_BBOX}) — if it's outside, detection drifted to a "
            "different region entirely (VLM mislocalization or CV merging with background)",
        )


if __name__ == "__main__":
    unittest.main()
