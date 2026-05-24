"""
Unit tests for vision.stack_frames.

Fixtures: tests/fixtures/move_forward/
  droidcam_000.jpg  (earliest — most transparent)
  droidcam_024.jpg  (middle)
  droidcam_036.jpg  (latest — least transparent / most opaque)

Output: tests/fixtures/move_forward/stacked_output.jpg  (written for manual inspection)
"""
import base64
import pathlib
import unittest

import cv2
import numpy as np

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "move_forward"
OUTPUT_PATH = FIXTURE_DIR / "stacked_output.jpg"


def _load_b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _decode_b64(b64: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)


class TestStackFrames(unittest.TestCase):

    def setUp(self):
        from mcp_robot.vision import stack_frames
        self._fn = stack_frames
        self._frames = [
            _load_b64(FIXTURE_DIR / "droidcam_000.jpg"),
            _load_b64(FIXTURE_DIR / "droidcam_024.jpg"),
            _load_b64(FIXTURE_DIR / "droidcam_036.jpg"),
        ]

    def test_output_is_valid_jpeg(self):
        result = self._fn(self._frames)
        img = _decode_b64(result)
        self.assertIsNotNone(img)

    def test_output_dimensions_match_input(self):
        first = _decode_b64(self._frames[0])
        result = self._fn(self._frames)
        out = _decode_b64(result)
        self.assertEqual(out.shape, first.shape)

    def test_single_frame_roundtrip(self):
        """Single frame stacked alone should produce a valid image of the same size."""
        single = [self._frames[0]]
        result = self._fn(single)
        img = _decode_b64(result)
        orig = _decode_b64(self._frames[0])
        self.assertEqual(img.shape, orig.shape)

    def test_later_frame_has_more_influence(self):
        """The last frame should have more weight than the first.

        Verify by computing the mean-squared difference of the composite
        against the last frame vs. against the first frame.
        """
        result_img = _decode_b64(self._fn(self._frames)).astype(float)
        first_img = _decode_b64(self._frames[0]).astype(float)
        last_img = _decode_b64(self._frames[-1]).astype(float)

        mse_to_first = float(np.mean((result_img - first_img) ** 2))
        mse_to_last = float(np.mean((result_img - last_img) ** 2))
        self.assertLess(mse_to_last, mse_to_first,
                        "Composite should be closer to the last (heaviest) frame")

    def test_raises_on_empty_input(self):
        with self.assertRaises(ValueError):
            self._fn([])

    def test_save_stacked_output(self):
        """Saves stacked_output.jpg to the fixture dir for manual inspection."""
        result = self._fn(self._frames)
        OUTPUT_PATH.write_bytes(base64.b64decode(result))
        self.assertTrue(OUTPUT_PATH.exists())


if __name__ == "__main__":
    unittest.main()
