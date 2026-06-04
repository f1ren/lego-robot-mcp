"""
Unit tests for mcp_robot.inpainting.remove_robot_and_target.

Test cases
----------
TestInpainting
    droidcam_west_heading.jpg — robot + blue cup target.

TestInpaintingRobotHidingSwitch
    droidcam_robot_hiding_switch.jpg — robot parked in front of a wall switch;
    no YOLO target (switch is not a COCO class).  The inpainted output should
    reveal the switch behind the robot.  Depth maps are run on both the original
    and inpainted frames and saved so you can evaluate whether the floor reads
    as a coherent plane after inpainting.

Outputs go to tests/fixtures/navigation/annotated/inpainting/<case>/
  mask.png             — white = pixels removed
  inpainted.jpg        — LaMa-filled result
  side_by_side.jpg     — original | mask | inpainted
  depth_original.jpg   — Depth Anything V2 heat-map of the raw frame
  depth_inpainted.jpg  — depth heat-map of the inpainted frame

Run with:
    python -m pytest tests/test_inpainting.py -v -s
"""
import pathlib
import unittest

import cv2
import numpy as np

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
INPAINT_BASE = FIXTURES / "navigation" / "annotated" / "inpainting"


def _load(path: pathlib.Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    assert bgr is not None, f"Could not load fixture: {path}"
    return bgr


def _depth_heatmap(bgr: np.ndarray) -> np.ndarray:
    """Run Depth Anything V2 and return an INFERNO colour-map image."""
    from mcp_robot.navigation import estimate_depth
    d = estimate_depth(bgr)
    d_norm = ((d - d.min()) / (d.max() - d.min() + 1e-8) * 255).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)


def _save_outputs(
    out_dir: pathlib.Path,
    bgr: np.ndarray,
    mask: np.ndarray,
    inpainted: np.ndarray,
    run_depth: bool = False,
) -> dict[str, pathlib.Path]:
    """Write evaluation images and optionally depth maps. Returns paths dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, pathlib.Path] = {}

    cv2.imwrite(str(out_dir / "mask.png"), mask)
    saved["mask"] = out_dir / "mask.png"

    cv2.imwrite(str(out_dir / "inpainted.jpg"), inpainted, [cv2.IMWRITE_JPEG_QUALITY, 92])
    saved["inpainted"] = out_dir / "inpainted.jpg"

    mask_vis = np.zeros_like(bgr)
    mask_vis[mask > 0] = (0, 200, 0)
    side = np.concatenate([bgr, mask_vis, inpainted], axis=1)
    cv2.imwrite(str(out_dir / "side_by_side.jpg"), side, [cv2.IMWRITE_JPEG_QUALITY, 88])
    saved["side_by_side"] = out_dir / "side_by_side.jpg"

    if run_depth:
        depth_orig = _depth_heatmap(bgr)
        cv2.imwrite(str(out_dir / "depth_original.jpg"), depth_orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved["depth_original"] = out_dir / "depth_original.jpg"

        depth_inp = _depth_heatmap(inpainted)
        cv2.imwrite(str(out_dir / "depth_inpainted.jpg"), depth_inp, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved["depth_inpainted"] = out_dir / "depth_inpainted.jpg"

    return saved


class TestInpainting(unittest.TestCase):
    """Robot + blue cup target (droidcam_west_heading.jpg)."""

    OUT_DIR = INPAINT_BASE / "cup"

    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import detect_heading
        from mcp_robot.grasp_readiness import _yolo_detect, _pick_target, _load_model
        from mcp_robot.inpainting import build_removal_mask, remove_robot_and_target

        bgr = _load(FIXTURES / "navigation" / "droidcam_west_heading.jpg")
        h_result = detect_heading(bgr)

        _load_model()
        objects = _yolo_detect(bgr, target_class="cup")
        target = (
            _pick_target(objects, h_result)
            if (objects and h_result)
            else (max(objects, key=lambda o: o.confidence) if objects else None)
        )

        print(f"\n[cup] Heading detected: {h_result is not None}")
        print(f"[cup] Target: {target.class_name if target else 'none'}")

        mask = build_removal_mask(bgr, h_result, target)
        inpainted, _ = remove_robot_and_target(bgr, h_result, target)

        saved = _save_outputs(cls.OUT_DIR, bgr, mask, inpainted, run_depth=False)
        print(f"[cup] Outputs: {list(saved.values())}")

        cls._bgr       = bgr
        cls._mask      = mask
        cls._inpainted = inpainted
        cls._target    = target
        cls._saved     = saved

    # ── mask sanity ───────────────────────────────────────────────────────────

    def test_mask_same_shape_as_input(self):
        h, w = self._bgr.shape[:2]
        self.assertEqual(self._mask.shape, (h, w))

    def test_mask_has_robot_region(self):
        frac = (self._mask > 0).mean()
        self.assertGreater(frac, 0.01, f"Mask covers only {frac:.1%} — robot not masked")

    def test_mask_not_entire_frame(self):
        frac = (self._mask > 0).mean()
        self.assertLess(frac, 0.60, f"Mask covers {frac:.1%} — too aggressive")

    def test_mask_covers_target_if_detected(self):
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
        ys, xs = np.where(self._mask > 0)
        if len(ys) == 0:
            self.skipTest("No masked pixels")
        mean_diff = float(np.abs(
            self._bgr[ys, xs].astype(np.float32) - self._inpainted[ys, xs].astype(np.float32)
        ).mean())
        self.assertGreater(mean_diff, 5.0, f"Inpainting had no effect (mean diff={mean_diff:.1f})")

    def test_unmasked_region_preserved(self):
        ys, xs = np.where(self._mask == 0)
        if len(ys) == 0:
            self.skipTest("All pixels masked")
        mean_diff = float(np.abs(
            self._bgr[ys, xs].astype(np.float32) - self._inpainted[ys, xs].astype(np.float32)
        ).mean())
        self.assertLess(mean_diff, 10.0, f"Background corrupted (mean diff={mean_diff:.1f})")

    # ── output files written ──────────────────────────────────────────────────

    def test_output_files_written(self):
        min_sizes = {"mask": 500, "inpainted": 10_000, "side_by_side": 20_000}
        for key, min_bytes in min_sizes.items():
            p = self._saved[key]
            self.assertTrue(p.exists(), f"Not written: {p}")
            self.assertGreater(p.stat().st_size, min_bytes,
                               f"Too small ({p.stat().st_size} B): {p}")


class TestInpaintingRobotHidingSwitch(unittest.TestCase):
    """Robot in front of a wall switch (droidcam_robot_hiding_switch.jpg).

    No YOLO target — switch is not a COCO class.  Only the robot is removed.
    Depth maps are saved for both the original and inpainted frame so you can
    evaluate whether the floor plane looks coherent after inpainting.
    """

    OUT_DIR = INPAINT_BASE / "robot_hiding_switch"

    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import detect_heading
        from mcp_robot.grasp_readiness import _yolo_detect, _pick_target, _load_model
        from mcp_robot.inpainting import build_removal_mask, remove_robot_and_target

        bgr = _load(FIXTURES / "navigation" / "droidcam_robot_hiding_switch.jpg")
        h_result = detect_heading(bgr)

        _load_model()
        objects = _yolo_detect(bgr, target_class="cup")
        target = (
            _pick_target(objects, h_result)
            if (objects and h_result)
            else (max(objects, key=lambda o: o.confidence) if objects else None)
        )

        print(f"\n[switch] Heading detected: {h_result is not None}")
        if h_result:
            print(f"[switch] Forward: {h_result.forward}  body_center: {h_result.body_center}")
        print(f"[switch] Target: {target.class_name + f' conf={target.confidence:.2f}' if target else 'none'}")

        mask = build_removal_mask(bgr, h_result, target)
        inpainted, _ = remove_robot_and_target(bgr, h_result, target)

        saved = _save_outputs(cls.OUT_DIR, bgr, mask, inpainted, run_depth=True)
        print(f"[switch] Outputs: {list(saved.values())}")

        cls._bgr       = bgr
        cls._mask      = mask
        cls._inpainted = inpainted
        cls._heading   = h_result
        cls._target    = target
        cls._saved     = saved

    # ── heading ───────────────────────────────────────────────────────────────

    def test_robot_detected(self):
        self.assertIsNotNone(self._heading, "Yellow robot body not found in frame")

    # ── mask sanity ───────────────────────────────────────────────────────────

    def test_mask_same_shape_as_input(self):
        h, w = self._bgr.shape[:2]
        self.assertEqual(self._mask.shape, (h, w))

    def test_cup_detected(self):
        self.assertIsNotNone(self._target, "YOLO did not detect the cup in this frame")

    def test_mask_covers_robot(self):
        frac = (self._mask > 0).mean()
        self.assertGreater(frac, 0.01, f"Mask covers only {frac:.1%} — robot not masked")

    def test_mask_covers_cup_if_detected(self):
        if self._target is not None:
            cx = (self._target.x1 + self._target.x2) // 2
            cy = (self._target.y1 + self._target.y2) // 2
            self.assertEqual(self._mask[cy, cx], 255,
                             f"Cup centre ({cx},{cy}) not in mask")

    def test_mask_not_entire_frame(self):
        frac = (self._mask > 0).mean()
        self.assertLess(frac, 0.60, f"Mask covers {frac:.1%} — too aggressive")

    # ── inpainting output sanity ──────────────────────────────────────────────

    def test_inpainted_same_shape_as_input(self):
        self.assertEqual(self._inpainted.shape, self._bgr.shape)

    def test_inpainted_is_valid_uint8(self):
        self.assertEqual(self._inpainted.dtype, np.uint8)
        self.assertGreaterEqual(int(self._inpainted.min()), 0)
        self.assertLessEqual(int(self._inpainted.max()), 255)

    def test_masked_region_changed(self):
        ys, xs = np.where(self._mask > 0)
        if len(ys) == 0:
            self.skipTest("No masked pixels")
        mean_diff = float(np.abs(
            self._bgr[ys, xs].astype(np.float32) - self._inpainted[ys, xs].astype(np.float32)
        ).mean())
        self.assertGreater(mean_diff, 5.0, f"Inpainting had no effect (mean diff={mean_diff:.1f})")

    def test_unmasked_region_preserved(self):
        ys, xs = np.where(self._mask == 0)
        if len(ys) == 0:
            self.skipTest("All pixels masked")
        mean_diff = float(np.abs(
            self._bgr[ys, xs].astype(np.float32) - self._inpainted[ys, xs].astype(np.float32)
        ).mean())
        self.assertLess(mean_diff, 10.0, f"Background corrupted (mean diff={mean_diff:.1f})")

    # ── output files written (includes depth maps) ────────────────────────────

    def test_output_files_written(self):
        min_sizes = {
            "mask": 500, "inpainted": 10_000, "side_by_side": 20_000,
            "depth_original": 10_000, "depth_inpainted": 10_000,
        }
        for key, min_bytes in min_sizes.items():
            p = self._saved[key]
            self.assertTrue(p.exists(), f"Not written: {p}")
            self.assertGreater(p.stat().st_size, min_bytes,
                               f"Too small ({p.stat().st_size} B): {p}")


if __name__ == "__main__":
    unittest.main()
