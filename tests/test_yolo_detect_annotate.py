"""
Visualise _yolo_detect results on droidcam_cup.jpg fixture image.

Saves an annotated copy to tests/fixtures/grasp_readiness/annotated/ for manual
inspection.  Run with:

    python -m pytest tests/test_yolo_detect_annotate.py -s
"""
import pathlib
import unittest

import cv2

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "grasp_readiness"
OUT_DIR  = FIXTURES / "annotated"
CUP_IMG  = FIXTURES / "droidcam_cup.jpg"


def _annotate(bgr, objects):
    out = bgr.copy()
    h, w = out.shape[:2]
    for obj in objects:
        cv2.rectangle(out, (obj.x1, obj.y1), (obj.x2, obj.y2), (0, 255, 0), 2)
        cx, cy = obj.center
        cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
        label = f"{obj.class_name} {obj.confidence:.2f}"
        cv2.putText(out, label, (obj.x1, max(obj.y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    count_txt = f"YOLO detections: {len(objects)}"
    cv2.putText(out, count_txt, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
    return out


class TestYoloDetectAnnotate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mcp_robot.grasp_readiness import _yolo_detect, _load_model
        _load_model()
        cls._yolo_detect = staticmethod(_yolo_detect)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_annotate_cup_image(self):
        self.assertTrue(CUP_IMG.exists(), f"Fixture not found: {CUP_IMG}")
        bgr = cv2.imread(str(CUP_IMG))
        self.assertIsNotNone(bgr, f"Could not load {CUP_IMG}")

        objects = self._yolo_detect(bgr)
        print(f"\ndroidcam_cup.jpg: {len(objects)} YOLO detection(s)")
        for o in objects:
            print(f"  class={o.class_name!r}  conf={o.confidence:.2f}"
                  f"  center={o.center}  bbox=({o.x1},{o.y1},{o.x2},{o.y2})")

        annotated = _annotate(bgr, objects)
        out_path = OUT_DIR / "droidcam_cup_yolo.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"  -> saved {out_path}")


if __name__ == "__main__":
    unittest.main()
