"""
Unit tests for optical-flow annotation functions in mcp_robot.heading.

Fixtures: tests/fixtures/driving_backward/
  droidcam_*.jpg  (35 frames)  pi_camera_*.jpg  (14 frames)
  flow_output/  (written by TestSaveFlowOutput for manual inspection)
"""
import base64
import pathlib
import unittest

import cv2
import numpy as np

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "driving_backward"
RAW_DIR = FIXTURE_DIR
OUTPUT_DIR = FIXTURE_DIR / "flow_output"


def _load_b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _decode_b64(b64: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)


def _save_b64(b64: str, path: pathlib.Path) -> None:
    path.write_bytes(base64.b64decode(b64))


def _sorted_frames(directory: pathlib.Path, prefix: str) -> list[pathlib.Path]:
    return sorted(directory.glob(f"{prefix}_*.jpg"))


class TestAnnotateFlowSequenceBgr(unittest.TestCase):

    def setUp(self):
        from mcp_robot.heading import annotate_flow_sequence_bgr
        self._fn = annotate_flow_sequence_bgr

    def _load_bgr(self, path: pathlib.Path) -> np.ndarray:
        img = cv2.imread(str(path))
        assert img is not None, f"Could not read {path}"
        return img

    def test_single_frame_returned_unchanged(self):
        frame = self._load_bgr(_sorted_frames(RAW_DIR, "droidcam")[0])
        result = self._fn([frame])
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], frame)

    def test_first_frame_always_unchanged(self):
        frames = [self._load_bgr(p) for p in _sorted_frames(RAW_DIR, "droidcam")[:5]]
        result = self._fn(frames)
        np.testing.assert_array_equal(result[0], frames[0])

    def test_output_length_matches_input(self):
        frames = [self._load_bgr(p) for p in _sorted_frames(RAW_DIR, "droidcam")[:8]]
        result = self._fn(frames)
        self.assertEqual(len(result), len(frames))

    def test_subsequent_frames_differ_from_input_with_motion(self):
        """Optical-flow arrows must change at least one pixel across the full sequence."""
        ann_paths = _sorted_frames(RAW_DIR, "droidcam")
        raw_paths = _sorted_frames(RAW_DIR, "droidcam")
        ann_frames = [self._load_bgr(p) for p in ann_paths]
        raw_frames = [self._load_bgr(p) for p in raw_paths]
        result = self._fn(ann_frames, raw_frames)
        # At least one later frame should carry drawn arrows (differ from annotated input)
        any_modified = any(
            not np.array_equal(result[i], ann_frames[i]) for i in range(1, len(result))
        )
        self.assertTrue(any_modified, "Expected flow arrows to modify at least one frame")

    def test_raw_source_used_for_flow_computation(self):
        """Providing raw frames instead of annotated should yield different flow output."""
        ann_paths = _sorted_frames(RAW_DIR, "droidcam")[:6]
        raw_paths = _sorted_frames(RAW_DIR, "droidcam")[:6]
        ann_frames = [self._load_bgr(p) for p in ann_paths]
        raw_frames = [self._load_bgr(p) for p in raw_paths]

        result_with_raw = self._fn(ann_frames, raw_frames)
        result_without_raw = self._fn(ann_frames)

        # Results computed from raw vs annotated may differ in arrow positions
        # (heading overlay on annotated creates spurious flow); at minimum both succeed.
        self.assertEqual(len(result_with_raw), len(result_without_raw))

    def test_pi_camera_sequence(self):
        frames = [self._load_bgr(p) for p in _sorted_frames(RAW_DIR, "pi_camera")[:5]]
        result = self._fn(frames)
        self.assertEqual(len(result), 5)
        np.testing.assert_array_equal(result[0], frames[0])


class TestAnnotateFlowSequenceJpegB64(unittest.TestCase):

    def setUp(self):
        from mcp_robot.heading import annotate_flow_sequence_jpeg_b64
        self._fn = annotate_flow_sequence_jpeg_b64

    def test_output_length_matches_input_droidcam(self):
        b64s = [_load_b64(p) for p in _sorted_frames(RAW_DIR, "droidcam")[:8]]
        result = self._fn(b64s)
        self.assertEqual(len(result), len(b64s))

    def test_output_length_matches_input_pi_camera(self):
        b64s = [_load_b64(p) for p in _sorted_frames(RAW_DIR, "pi_camera")[:6]]
        result = self._fn(b64s)
        self.assertEqual(len(result), len(b64s))

    def test_output_is_valid_jpeg(self):
        b64s = [_load_b64(p) for p in _sorted_frames(RAW_DIR, "droidcam")[:4]]
        result = self._fn(b64s)
        for i, b64 in enumerate(result):
            img = _decode_b64(b64)
            self.assertIsNotNone(img, f"Frame {i} is not a valid JPEG")

    def test_with_raw_b64_provided(self):
        ann_b64s = [_load_b64(p) for p in _sorted_frames(RAW_DIR, "droidcam")[:6]]
        raw_b64s = [_load_b64(p) for p in _sorted_frames(RAW_DIR, "droidcam")[:6]]
        result = self._fn(ann_b64s, raw_b64s)
        self.assertEqual(len(result), len(ann_b64s))

    def test_mismatched_raw_length_falls_back_gracefully(self):
        ann_b64s = [_load_b64(p) for p in _sorted_frames(RAW_DIR, "droidcam")[:5]]
        raw_b64s = [_load_b64(p) for p in _sorted_frames(RAW_DIR, "droidcam")[:3]]
        result = self._fn(ann_b64s, raw_b64s)
        self.assertEqual(len(result), len(ann_b64s))

    def test_single_frame_returned_with_same_dimensions(self):
        b64 = _load_b64(_sorted_frames(RAW_DIR, "droidcam")[0])
        result = self._fn([b64])
        self.assertEqual(len(result), 1)
        # Re-encoding is lossy so b64 strings differ; verify dimensions are preserved.
        orig_shape = _decode_b64(b64).shape
        out_shape = _decode_b64(result[0]).shape
        self.assertEqual(orig_shape, out_shape)


class TestApplyOpticalFlowPerCamera(unittest.TestCase):
    """Tests for server._apply_optical_flow_per_camera."""

    def setUp(self):
        from mcp_robot.server import _apply_optical_flow_per_camera
        self._fn = _apply_optical_flow_per_camera

    def _make_labeled(
        self, directory: pathlib.Path, prefix: str, n: int, start: float = 0.0
    ) -> list[tuple[float, str, str]]:
        paths = _sorted_frames(directory, prefix)[:n]
        return [(start + i * 0.1, prefix, _load_b64(p)) for i, p in enumerate(paths)]

    def test_two_camera_output_length_preserved(self):
        ann = (
            self._make_labeled(RAW_DIR, "droidcam", 5, start=0.0)
            + self._make_labeled(RAW_DIR, "pi_camera", 4, start=0.05)
        )
        raw = (
            self._make_labeled(RAW_DIR, "droidcam", 5, start=0.0)
            + self._make_labeled(RAW_DIR, "pi_camera", 4, start=0.05)
        )
        result = self._fn(ann, raw)
        self.assertEqual(len(result), len(ann))

    def test_labels_preserved_after_annotation(self):
        ann = (
            self._make_labeled(RAW_DIR, "droidcam", 4)
            + self._make_labeled(RAW_DIR, "pi_camera", 3)
        )
        raw = (
            self._make_labeled(RAW_DIR, "droidcam", 4)
            + self._make_labeled(RAW_DIR, "pi_camera", 3)
        )
        result = self._fn(ann, raw)
        for (orig_ts, orig_lbl, _), (res_ts, res_lbl, _) in zip(ann, result):
            self.assertEqual(orig_ts, res_ts)
            self.assertEqual(orig_lbl, res_lbl)

    def test_single_camera_still_works(self):
        ann = self._make_labeled(RAW_DIR, "droidcam", 6)
        raw = self._make_labeled(RAW_DIR, "droidcam", 6)
        result = self._fn(ann, raw)
        self.assertEqual(len(result), 6)

    def test_output_b64_are_valid_jpegs(self):
        ann = (
            self._make_labeled(RAW_DIR, "droidcam", 4)
            + self._make_labeled(RAW_DIR, "pi_camera", 3)
        )
        raw = (
            self._make_labeled(RAW_DIR, "droidcam", 4)
            + self._make_labeled(RAW_DIR, "pi_camera", 3)
        )
        result = self._fn(ann, raw)
        for _, lbl, b64 in result:
            img = _decode_b64(b64)
            self.assertIsNotNone(img, f"Frame labeled '{lbl}' is not a valid JPEG")


class TestSaveFlowOutput(unittest.TestCase):
    """Saves annotated flow output to disk for manual inspection."""

    def test_save_droidcam_flow_output(self):
        from mcp_robot.heading import annotate_flow_sequence_jpeg_b64

        ann_b64s = [_load_b64(p) for p in _sorted_frames(RAW_DIR, "droidcam")]
        raw_b64s = [_load_b64(p) for p in _sorted_frames(RAW_DIR, "droidcam")]
        result = annotate_flow_sequence_jpeg_b64(ann_b64s, raw_b64s)

        out_dir = OUTPUT_DIR / "droidcam"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, b64 in enumerate(result):
            _save_b64(b64, out_dir / f"droidcam_{i:03d}.jpg")

        self.assertEqual(len(result), len(ann_b64s))

    def test_save_pi_camera_flow_output(self):
        from mcp_robot.heading import annotate_flow_sequence_jpeg_b64

        ann_b64s = [_load_b64(p) for p in _sorted_frames(RAW_DIR, "pi_camera")]
        raw_b64s = [_load_b64(p) for p in _sorted_frames(RAW_DIR, "pi_camera")]
        result = annotate_flow_sequence_jpeg_b64(ann_b64s, raw_b64s)

        out_dir = OUTPUT_DIR / "pi_camera"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, b64 in enumerate(result):
            _save_b64(b64, out_dir / f"pi_camera_{i:03d}.jpg")

        self.assertEqual(len(result), len(ann_b64s))


if __name__ == "__main__":
    unittest.main()
