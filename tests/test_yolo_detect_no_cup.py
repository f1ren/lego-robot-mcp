"""
Diagnose why YOLO fails to find a cup in tests/fixtures/grasp_readiness/droidcam_no_cup.png.

Runs YOLO at progressively lower confidence thresholds to see what the model
actually detects (and at what confidence), then saves an annotated image for
manual inspection.

Run with:
    python -m pytest tests/test_yolo_detect_no_cup.py -s
"""
import pathlib
import unittest

import cv2
import numpy as np

FIXTURES   = pathlib.Path(__file__).parent / "fixtures" / "grasp_readiness"
OUT_DIR    = FIXTURES / "annotated"
TARGET_IMG = FIXTURES / "droidcam_no_cup.png"

# Thresholds to probe — start at production value, step down to near-zero
_PROBE_CONFS = [0.75, 0.50, 0.30, 0.10, 0.05]
_YOLO_IOU    = 0.45
_CUP_SYNONYMS = frozenset({"cup", "bowl", "bottle", "vase"})


def _annotate(bgr: np.ndarray, results, conf_threshold: float) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    detections = 0
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if cls_name in _CUP_SYNONYMS else (128, 128, 128)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            detections += 1
    header = f"conf>={conf_threshold:.2f}  detections={detections}"
    cv2.putText(out, header, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    return out


class TestYoloNoCup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ultralytics import YOLO
        cls.model = YOLO("yolo11n.pt")
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_probe_confidence_thresholds(self):
        self.assertTrue(TARGET_IMG.exists(), f"Fixture not found: {TARGET_IMG}")
        bgr = cv2.imread(str(TARGET_IMG))
        self.assertIsNotNone(bgr, f"Could not load {TARGET_IMG}")
        print(f"\nImage size: {bgr.shape[1]}x{bgr.shape[0]}")

        found_cup_at = None
        for conf in _PROBE_CONFS:
            results = self.model(bgr, conf=conf, iou=_YOLO_IOU, verbose=False)
            cup_hits = []
            all_hits = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = r.names[cls_id]
                    c = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    all_hits.append((cls_name, c, x1, y1, x2, y2))
                    if cls_name in _CUP_SYNONYMS:
                        cup_hits.append((cls_name, c, x1, y1, x2, y2))

            print(f"\n  conf>={conf:.2f}: {len(all_hits)} total detections")
            for cls_name, c, x1, y1, x2, y2 in all_hits:
                marker = " *** CUP ***" if cls_name in _CUP_SYNONYMS else ""
                print(f"    {cls_name:<20} conf={c:.3f}  bbox=({x1},{y1},{x2},{y2}){marker}")

            if cup_hits and found_cup_at is None:
                found_cup_at = conf
                print(f"  --> Cup first detected at conf>={conf:.2f}")

            annotated = _annotate(bgr, results, conf)
            out_path = OUT_DIR / f"droidcam_no_cup_conf{int(conf*100):03d}.jpg"
            cv2.imwrite(str(out_path), annotated)

        if found_cup_at is None:
            print("\n  RESULT: Cup NOT detected at any confidence threshold — "
                  "image may not contain a detectable cup for YOLO.")
        else:
            print(f"\n  RESULT: Cup detected starting at conf>={found_cup_at:.2f} "
                  f"(production threshold is 0.75).")
        print(f"\nAnnotated images saved to: {OUT_DIR}")


class TestFullPipelineNoCup(unittest.TestCase):
    """Run the full check_grasp_readiness pipeline on the same image."""

    def test_grasp_readiness_result(self):
        self.assertTrue(TARGET_IMG.exists(), f"Fixture not found: {TARGET_IMG}")

        # Load as BGR (drop alpha if RGBA) — same as production
        bgr = cv2.imread(str(TARGET_IMG))
        self.assertIsNotNone(bgr, f"Could not load {TARGET_IMG}")
        print(f"\nImage: {TARGET_IMG.name}  shape={bgr.shape}")

        from mcp_robot.grasp_readiness import check_grasp_readiness, _load_model, annotate_frame_with_object
        _load_model()

        result = check_grasp_readiness(bgr)
        print(f"\ncheck_grasp_readiness result:")
        print(f"  ready            = {result.ready}")
        print(f"  object_detected  = {result.object_detected}")
        print(f"  touches_body     = {result.touches_body}")
        print(f"  arrow_well_over  = {result.arrow_well_over}")
        print(f"  perp_dist_px     = {result.perp_dist_px:.1f}")
        print(f"  dist_to_front_px = {result.dist_to_front_px:.1f}")
        print(f"  reason           = {result.reason}")
        if result.object_class:
            print(f"  object_class     = {result.object_class}  conf={result.object_confidence:.2f}")
        print(f"\n  Full dict: {result.to_dict()}")

        annotated = annotate_frame_with_object(bgr)
        out_path = OUT_DIR / "droidcam_no_cup_full_pipeline.jpg"
        if isinstance(annotated, np.ndarray):
            cv2.imwrite(str(out_path), annotated)
            print(f"\n  -> annotated frame saved to {out_path}")
        else:
            print(f"\n  WARNING: annotate_frame_with_object returned {type(annotated)} — skipping save")


if __name__ == "__main__":
    unittest.main()
