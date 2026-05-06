"""
Unit tests for vision._has_motion.
"""
import base64
import pathlib
import unittest

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "static_video"


def _load_b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


class TestHasMotion(unittest.TestCase):
    def setUp(self):
        from mcp_robot.vision import _has_motion
        self._has_motion = _has_motion

    def _frames(self, *names: str) -> list[str]:
        return [_load_b64(FIXTURES / n) for n in names]

    def test_static_video_returns_false(self):
        frames = sorted(FIXTURES.glob("droidcam_*.jpg"))
        b64_frames = [_load_b64(f) for f in frames]
        self.assertFalse(
            self._has_motion(b64_frames),
            "Expected no motion in a static video clip",
        )

    def test_single_frame_returns_true(self):
        frames = sorted(FIXTURES.glob("droidcam_*.jpg"))
        self.assertTrue(self._has_motion([_load_b64(frames[0])]))

    def test_empty_returns_true(self):
        self.assertTrue(self._has_motion([]))

    def test_synthetic_motion_returns_true(self):
        import cv2
        import numpy as np

        img_a = np.zeros((100, 100), dtype=np.uint8)
        img_b = np.full((100, 100), 50, dtype=np.uint8)

        def _encode(arr):
            _, buf = cv2.imencode(".jpg", arr)
            return base64.b64encode(buf.tobytes()).decode()

        self.assertTrue(self._has_motion([_encode(img_a), _encode(img_b)]))

    def test_identical_frames_return_false(self):
        frames = sorted(FIXTURES.glob("droidcam_*.jpg"))
        b64 = _load_b64(frames[0])
        self.assertFalse(self._has_motion([b64, b64]))


if __name__ == "__main__":
    unittest.main()
