"""
Regression test for locate_object_hybrid's cv_refine_location background-merge
fix (see mcp_robot/vision.py::_CV_SEARCH_RADII).

Saves an annotated copy to tests/fixtures/grasp_readiness/annotated/ for
manual inspection.  Run with:

    python -m pytest tests/test_hybrid_locate.py -s
"""
import pathlib
import unittest

import cv2
import pytest

FIXTURES  = pathlib.Path(__file__).parent / "fixtures" / "grasp_readiness"
OUT_DIR   = FIXTURES / "annotated"

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

        vlm = self.vision.locate_object_vlm(bgr, description)
        if isinstance(vlm, self.vision.LowConfidenceDetection):
            self.skipTest(f"{img_path.name}: {vlm}")
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


class TestCvRefineLocationFallback(unittest.TestCase):
    """Regression test for the zero-overlap last-resort branch in
    cv_refine_location (mcp_robot/vision.py) — distinct from the background-merge
    fix above.  Feeds the exact VLM output logged in production on
    light_switch_cabinet.jpg directly to cv_refine_location, so this runs
    offline (no live VLM call, not marked `live`).

    On 2026-07-18 the VLM correctly placed rough_bbox=[525,637,648,700] on a
    wall light switch, but its proposed HSV range (warm hue, near-zero
    saturation floor) also matches this frame's sunlit wood floor/panel. No
    contour anywhere in the search region satisfied the area band, nor even
    merely overlapped the rough bbox — cv_refine_location used to fall back to
    the single largest raw blob in the widest window with no positional
    constraint at all, landing on a 50k-pixel patch of floor next to the robot
    instead of the switch. It must now return None so locate_object_hybrid
    falls back to the VLM's (correct) rough bbox instead.
    """

    FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "grasp_readiness" / "light_switch_cabinet.jpg"

    # Exact values logged by locate_object_vlm for this frame — replayed here
    # instead of calling Gemini again so this test is deterministic and free.
    _ROUGH_BBOX = (525, 637, 648, 700)
    _HSV_LO = (15, 10, 200)
    _HSV_HI = (40, 255, 255)
    _AREA_FRAC = 0.0080

    def test_no_overlapping_contour_returns_none_not_wrong_blob(self):
        import numpy as np

        from mcp_robot import vision

        bgr = cv2.imread(str(self.FIXTURE))
        self.assertIsNotNone(bgr, f"Could not load {self.FIXTURE}")

        hsv_lo = np.array(self._HSV_LO, dtype=np.uint8)
        hsv_hi = np.array(self._HSV_HI, dtype=np.uint8)
        refined = vision.cv_refine_location(bgr, self._ROUGH_BBOX, hsv_lo, hsv_hi, self._AREA_FRAC)
        self.assertIsNone(
            refined,
            f"cv_refine_location should give up (None) when no contour overlaps the rough "
            f"bbox anywhere, not confidently return an unrelated blob; got {refined}",
        )


if __name__ == "__main__":
    unittest.main()
