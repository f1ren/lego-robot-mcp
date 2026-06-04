"""
Unit tests for mcp_robot.inpainting.remove_robot_and_target.

Uses tests/fixtures/navigation/droidcam_west_heading.jpg — a DroidCam frame
with a visible yellow robot body and a blue cup target.

Outputs are saved to tests/fixtures/navigation/annotated/inpainting/ for
visual evaluation:
  mask.png          — white = pixels removed (robot + target regions)
  inpainted.jpg     — result after LaMa fills the masked regions
  side_by_side.jpg  — original | mask visualisation | inpainted, for easy comparison

Run with:
    python -m pytest tests/test_inpainting.py -v -s
"""
import pathlib
import unittest

import cv2
import numpy as np

FIXTURES  = pathlib.Path(__file__).parent / "fixtures"
IMG_PATH  = FIXTURES / "navigation" / "droidcam_west_heading.jpg"
OUT_DIR   = FIXTURES / "navigation" / "annotated" / "inpainting"


def _load(path: pathlib.Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    assert bgr is not None, f"Could not load fixture: {path}"
    return bgr


class TestInpainting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import detect_heading
        from mcp_robot.grasp_readiness import _yolo_detect, _pick_target, _load_model
        from mcp_robot.inpainting import build_removal_mask, remove_robot_and_target

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        bgr = _load(IMG_PATH)
        h_result = detect_heading(bgr)

        _load_model()
        objects = _yolo_detect(bgr, target_class="cup")
        target = (
            _pick_target(objects, h_result)
            if (objects and h_result)
            else (max(objects, key=lambda o: o.confidence) if objects else None)
        )

        print(f"\n[inpainting] Heading detected: {h_result is not None}")
        print(f"[inpainting] Target: {target.class_name if target else 'none'}")

        mask = build_removal_mask(bgr, h_result, target)
        inpainted, _ = remove_robot_and_target(bgr, h_result, target)

        # ── Save outputs for visual evaluation ───────────────────────────────
        cv2.imwrite(str(OUT_DIR / "mask.png"), mask)

        cv2.imwrite(str(OUT_DIR / "inpainted.jpg"), inpainted,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Side-by-side: original | mask (green/black) | inpainted
        mask_vis = np.zeros_like(bgr)
        mask_vis[mask > 0] = (0, 200, 0)
        side = np.concatenate([bgr, mask_vis, inpainted], axis=1)
        cv2.imwrite(str(OUT_DIR / "side_by_side.jpg"), side,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

        print(f"[inpainting] Outputs saved to: {OUT_DIR}")

        cls._bgr      = bgr
        cls._mask     = mask
        cls._inpainted = inpainted
        cls._heading  = h_result
        cls._target   = target

    # ── mask sanity ───────────────────────────────────────────────────────────

    def test_mask_same_shape_as_input(self):
        h, w = self._bgr.shape[:2]
        self.assertEqual(self._mask.shape, (h, w))

    def test_mask_has_robot_region(self):
        """Robot body is yellow — mask must cover some pixels."""
        frac = (self._mask > 0).mean()
        self.assertGreater(frac, 0.01, f"Mask covers only {frac:.1%} of frame — robot not masked")

    def test_mask_not_entire_frame(self):
        """Mask must leave most of the frame unmasked (floor should survive)."""
        frac = (self._mask > 0).mean()
        self.assertLess(frac, 0.60, f"Mask covers {frac:.1%} — too aggressive")

    def test_mask_covers_target_if_detected(self):
        """If a target was detected, its centre pixel must be in the mask."""
        if self._target is not None:
            cx = (self._target.x1 + self._target.x2) // 2
            cy = (self._target.y1 + self._target.y2) // 2
            self.assertEqual(self._mask[cy, cx], 255,
                             f"Target centre ({cx},{cy}) not in mask")

    # ── inpainting output sanity ──────────────────────────────────────────────

    def test_inpainted_same_shape_as_input(self):
        self.assertEqual(self._inpainted.shape, self._bgr.shape)

    def test_inpainted_is_valid_uint8(self):
        self.assertEqual(self._inpainted.dtype, np.uint8)
        self.assertGreaterEqual(int(self._inpainted.min()), 0)
        self.assertLessEqual(int(self._inpainted.max()), 255)

    def test_masked_region_changed(self):
        """Pixels inside the mask must differ from the original (inpainting happened)."""
        ys, xs = np.where(self._mask > 0)
        if len(ys) == 0:
            self.skipTest("No masked pixels — nothing to verify")
        orig_vals   = self._bgr[ys, xs].astype(np.float32)
        result_vals = self._inpainted[ys, xs].astype(np.float32)
        mean_diff = float(np.abs(orig_vals - result_vals).mean())
        self.assertGreater(mean_diff, 5.0,
                           f"Masked pixels barely changed (mean diff={mean_diff:.1f}) — inpainting may have failed")

    def test_unmasked_region_preserved(self):
        """Pixels outside the mask must be nearly identical to the original."""
        ys, xs = np.where(self._mask == 0)
        if len(ys) == 0:
            self.skipTest("All pixels masked — cannot check preservation")
        orig_vals   = self._bgr[ys, xs].astype(np.float32)
        result_vals = self._inpainted[ys, xs].astype(np.float32)
        mean_diff = float(np.abs(orig_vals - result_vals).mean())
        self.assertLess(mean_diff, 10.0,
                        f"Unmasked pixels changed too much (mean diff={mean_diff:.1f}) — inpainting corrupted background")

    # ── output files written ──────────────────────────────────────────────────

    def test_output_files_written(self):
        min_sizes = {"mask.png": 500, "inpainted.jpg": 10_000, "side_by_side.jpg": 20_000}
        for fname, min_bytes in min_sizes.items():
            p = OUT_DIR / fname
            self.assertTrue(p.exists(), f"Output not written: {p}")
            self.assertGreater(p.stat().st_size, min_bytes,
                               f"Suspiciously small ({p.stat().st_size} B): {p}")


if __name__ == "__main__":
    unittest.main()
