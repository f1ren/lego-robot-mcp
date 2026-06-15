"""
Unit tests for recorder.SegmentRecorder.
"""
import base64
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import cv2

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _load_b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _read_manifest(manifest_path: str) -> list[dict]:
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


class _FakeCache:
    """Minimal stand-in for _PiFrameCache/_DroidCamFrameCache.clip_since."""

    def __init__(self, frames=None):
        self.frames = frames or []

    def clip_since(self, t_start, max_fps=3.0):
        frames = [f for f in self.frames if f["ts"] >= t_start]
        return frames or None


class TestSegmentRecorder(unittest.TestCase):
    def setUp(self):
        from mcp_robot.recorder import SegmentRecorder
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.SegmentRecorder = SegmentRecorder

    def tearDown(self):
        self._tmp.cleanup()

    def _make_recorder(self, **overrides):
        kwargs = dict(
            segment_dir=self.tmpdir,
            manifest_path=os.path.join(self.tmpdir, "index.jsonl"),
            preroll_s=0.5,
            cooldown_s=1.0,
            fps_by_camera={"droidcam": 10.0},
        )
        kwargs.update(overrides)
        return self.SegmentRecorder(**kwargs)

    def _build_closed_gripper_segment(self, rec, t0=1000.0):
        """Feed 2 static frames + a gripper-motion trigger frame, then close
        the segment via cooldown by repeating the trigger frame.

        Returns (next_t, closed_segment).
        """
        static_b64 = [_load_b64(f) for f in sorted((FIXTURES / "static_video").glob("droidcam_*.jpg"))[:2]]
        gripper_b64 = _load_b64(FIXTURES / "gripper_motion" / "droidcam_008.jpg")

        t = t0
        rec.on_frame("droidcam", static_b64[0], t)
        t = round(t + 0.1, 3)
        rec.on_frame("droidcam", static_b64[1], t)
        t = round(t + 0.1, 3)
        trigger_ts = t
        rec.on_frame("droidcam", gripper_b64, trigger_ts)

        close_ts = round(trigger_ts + rec.cooldown_s, 3)
        rec.on_frame("droidcam", gripper_b64, close_ts)

        state = rec._cameras["droidcam"]
        seg = state.recent_closed[-1]
        return round(close_ts + 0.1, 3), seg

    # ── 1: no motion, no segment ────────────────────────────────────────────

    def test_static_frames_produce_no_segment(self):
        rec = self._make_recorder()
        frames = sorted((FIXTURES / "static_video").glob("droidcam_*.jpg"))
        t = 1000.0
        for f in frames:
            rec.on_frame("droidcam", _load_b64(f), t)
            t = round(t + 0.1, 3)

        state = rec._cameras.get("droidcam")
        self.assertIsNotNone(state)
        self.assertIsNone(state.open_segment)
        self.assertEqual(len(state.recent_closed), 0)
        self.assertFalse(os.path.exists(rec.manifest_path))

    # ── 2: motion opens a segment with pre-roll frames ──────────────────────

    def test_motion_start_opens_segment_with_preroll(self):
        rec = self._make_recorder()

        entries = []
        t = 1000.0
        for f in sorted((FIXTURES / "static_video").glob("droidcam_*.jpg"))[:6]:
            entries.append({"frame": _load_b64(f), "ts": t})
            t = round(t + 0.1, 3)
        for name in ("droidcam_008.jpg", "droidcam_009.jpg"):
            entries.append({"frame": _load_b64(FIXTURES / "gripper_motion" / name), "ts": t})
            t = round(t + 0.1, 3)

        cache = _FakeCache(entries)
        for e in entries:
            rec.on_frame("droidcam", e["frame"], e["ts"], cache=cache)

        state = rec._cameras["droidcam"]
        seg = state.open_segment
        self.assertIsNotNone(seg)
        # trigger = gripper_008 (ts=1000.6); pre-roll = static frames with
        # ts < 1000.6 and ts >= trigger_ts - preroll_s (1000.1) -> 001..005 (5 frames)
        self.assertAlmostEqual(seg.start_ts, 1000.6, places=3)
        self.assertEqual(seg.frame_count, 7)

    # ── 3: active segment writes every frame, motion or not ─────────────────

    def test_active_segment_writes_all_frames_regardless_of_diff(self):
        rec = self._make_recorder()
        static_b64 = [_load_b64(f) for f in sorted((FIXTURES / "static_video").glob("droidcam_*.jpg"))[:2]]
        gripper_b64 = _load_b64(FIXTURES / "gripper_motion" / "droidcam_008.jpg")

        t = 1000.0
        rec.on_frame("droidcam", static_b64[0], t)
        t = round(t + 0.1, 3)
        rec.on_frame("droidcam", static_b64[1], t)
        t = round(t + 0.1, 3)
        rec.on_frame("droidcam", gripper_b64, t)  # cross-fixture diff -> motion, opens segment

        seg = rec._cameras["droidcam"].open_segment
        self.assertIsNotNone(seg)
        count_after_open = seg.frame_count

        t = round(t + 0.1, 3)
        rec.on_frame("droidcam", gripper_b64, t)  # identical frame -> no motion, still written

        self.assertEqual(seg.frame_count, count_after_open + 1)
        self.assertIsNotNone(rec._cameras["droidcam"].open_segment)

    # ── 4: cooldown closes and finalizes the segment ────────────────────────

    def test_cooldown_closes_segment(self):
        rec = self._make_recorder()
        _, seg = self._build_closed_gripper_segment(rec)

        state = rec._cameras["droidcam"]
        self.assertIsNone(state.open_segment)
        self.assertEqual(len(state.recent_closed), 1)
        self.assertEqual(seg.frame_count, 2)
        self.assertIsNotNone(seg.end_ts)

        self.assertEqual(_frame_count(seg.path), 2)

        records = _read_manifest(rec.manifest_path)
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0]["tool"])
        self.assertEqual(records[0]["frame_count"], 2)
        self.assertEqual(records[0]["path"], seg.path)

    # ── 5: new motion after a gap opens a new, distinct segment ─────────────

    def test_drive_motion_reopens_new_segment_after_gap(self):
        rec = self._make_recorder()
        t, seg1 = self._build_closed_gripper_segment(rec)

        drive005 = _load_b64(FIXTURES / "drive_motion" / "droidcam_005.jpg")
        drive006 = _load_b64(FIXTURES / "drive_motion" / "droidcam_006.jpg")
        rec.on_frame("droidcam", drive005, t)
        t = round(t + 0.1, 3)
        rec.on_frame("droidcam", drive006, t)

        state = rec._cameras["droidcam"]
        seg2 = state.open_segment
        self.assertIsNotNone(seg2)
        self.assertNotEqual(seg2.path, seg1.path)
        self.assertEqual(len(state.recent_closed), 1)

    # ── 6: tag_range on an open segment, flushed on close ───────────────────

    def test_tag_range_open_segment(self):
        rec = self._make_recorder()
        t, seg1 = self._build_closed_gripper_segment(rec)

        drive005 = _load_b64(FIXTURES / "drive_motion" / "droidcam_005.jpg")
        drive006 = _load_b64(FIXTURES / "drive_motion" / "droidcam_006.jpg")
        t1 = t
        rec.on_frame("droidcam", drive005, t1)
        t2 = round(t1 + 0.1, 3)
        rec.on_frame("droidcam", drive006, t2)

        ok = rec.tag_range("droidcam", t1, t2, {"tool": "drive", "change_description": "moved forward"})
        self.assertTrue(ok)

        seg2 = rec._cameras["droidcam"].open_segment
        self.assertIsNotNone(seg2)
        self.assertEqual(seg2.tool, "drive")
        self.assertFalse(seg2.flushed)

        close_ts = round(t2 + rec.cooldown_s, 3)
        rec.on_frame("droidcam", drive006, close_ts)  # identical -> closes seg2

        records = _read_manifest(rec.manifest_path)
        self.assertEqual(len(records), 2)
        rec2 = next(r for r in records if r["path"] == seg2.path)
        self.assertEqual(rec2["tool"], "drive")
        self.assertEqual(rec2["change_description"], "moved forward")

    # ── 7: tag_range on an already-closed segment rewrites its JSONL line ───

    def test_tag_range_closed_segment_rewrites_jsonl(self):
        rec = self._make_recorder()
        _, seg1 = self._build_closed_gripper_segment(rec)

        records_before = _read_manifest(rec.manifest_path)
        self.assertEqual(len(records_before), 1)

        ok = rec.tag_range("droidcam", seg1.start_ts, seg1.end_ts,
                            {"tool": "control_gripper", "change_description": "gripper opened"})
        self.assertTrue(ok)

        records_after = _read_manifest(rec.manifest_path)
        self.assertEqual(len(records_after), 1)
        self.assertEqual(records_after[0]["tool"], "control_gripper")
        self.assertEqual(records_after[0]["change_description"], "gripper opened")
        self.assertEqual(records_after[0]["path"], seg1.path)

    # ── 8: tag_range with no overlapping segment returns False ──────────────

    def test_tag_range_no_overlap_returns_false(self):
        rec = self._make_recorder()
        _, seg1 = self._build_closed_gripper_segment(rec)

        ok = rec.tag_range("droidcam", seg1.end_ts + 100, seg1.end_ts + 200, {"tool": "x"})
        self.assertFalse(ok)

        records = _read_manifest(rec.manifest_path)
        self.assertIsNone(records[0]["tool"])

    # ── 9: arm-motion fixtures behave the same as gripper-motion ────────────

    def test_arm_motion_segment(self):
        rec = self._make_recorder()

        entries = []
        t = 1000.0
        for f in sorted((FIXTURES / "static_video").glob("droidcam_*.jpg"))[:6]:
            entries.append({"frame": _load_b64(f), "ts": t})
            t = round(t + 0.1, 3)
        for name in ("droidcam_004.jpg", "droidcam_005.jpg"):
            entries.append({"frame": _load_b64(FIXTURES / "arm_motion" / name), "ts": t})
            t = round(t + 0.1, 3)

        cache = _FakeCache(entries)
        for e in entries:
            rec.on_frame("droidcam", e["frame"], e["ts"], cache=cache)

        seg = rec._cameras["droidcam"].open_segment
        self.assertIsNotNone(seg)
        self.assertEqual(seg.frame_count, 7)

    # ── 10: close() force-flushes an open segment ───────────────────────────

    def test_close_flushes_open_segment(self):
        rec = self._make_recorder()
        static_b64 = [_load_b64(f) for f in sorted((FIXTURES / "static_video").glob("droidcam_*.jpg"))[:2]]
        gripper_b64 = _load_b64(FIXTURES / "gripper_motion" / "droidcam_008.jpg")

        t = 1000.0
        rec.on_frame("droidcam", static_b64[0], t)
        t = round(t + 0.1, 3)
        rec.on_frame("droidcam", static_b64[1], t)
        t = round(t + 0.1, 3)
        rec.on_frame("droidcam", gripper_b64, t)

        seg = rec._cameras["droidcam"].open_segment
        self.assertIsNotNone(seg)

        rec.close()

        state = rec._cameras["droidcam"]
        self.assertIsNone(state.open_segment)
        self.assertEqual(len(state.recent_closed), 1)
        self.assertIsNotNone(seg.end_ts)
        self.assertEqual(_frame_count(seg.path), 1)

        records = _read_manifest(rec.manifest_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["end_ts"], round(seg.end_ts, 3))

    # ── 11: first frame for a fresh camera never triggers a segment ─────────

    def test_first_frame_never_triggers_segment(self):
        rec = self._make_recorder()
        gripper_b64 = _load_b64(FIXTURES / "gripper_motion" / "droidcam_008.jpg")
        rec.on_frame("droidcam", gripper_b64, 1000.0)

        state = rec._cameras["droidcam"]
        self.assertIsNone(state.open_segment)
        self.assertFalse(os.path.exists(rec.manifest_path))

    # ── 12: compile_task_video concatenates closed segments ─────────────────

    def test_concat_compile(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")

        rec = self._make_recorder()
        t, seg1 = self._build_closed_gripper_segment(rec)

        drive005 = _load_b64(FIXTURES / "drive_motion" / "droidcam_005.jpg")
        drive006 = _load_b64(FIXTURES / "drive_motion" / "droidcam_006.jpg")
        rec.on_frame("droidcam", drive005, t)
        t = round(t + 0.1, 3)
        rec.on_frame("droidcam", drive006, t)
        close_ts = round(t + rec.cooldown_s, 3)
        rec.on_frame("droidcam", drive006, close_ts)

        state = rec._cameras["droidcam"]
        self.assertEqual(len(state.recent_closed), 2)
        seg2 = state.recent_closed[-1]

        from mcp_robot.video_compiler import compile_task_video
        result = compile_task_video(
            since=str(seg1.start_ts - 1),
            manifest_path=rec.manifest_path,
            camera="droidcam",
            out_dir=self.tmpdir,
        )
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.segment_count, 2)
        self.assertEqual(_frame_count(result.video_path), seg1.frame_count + seg2.frame_count)

    # ── 13: closing remuxes the container fps to the measured rate ──────────

    def test_close_remuxes_to_actual_fps(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")

        rec = self._make_recorder()  # declared droidcam fps = 10.0
        _, seg = self._build_closed_gripper_segment(rec)  # 2 frames, 1.0s apart -> 1.0fps actual

        # cv2's CAP_PROP_FPS reads avg_frame_rate, which ffmpeg's duration
        # heuristics skew for very short (2-frame) clips; r_frame_rate is the
        # value _maybe_remux's itsscale directly targets and lands on 1/1.
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,nb_frames",
             "-of", "default=noprint_wrappers=1", seg.path],
            capture_output=True, text=True,
        )
        info = dict(line.split("=", 1) for line in out.stdout.strip().splitlines())
        num, den = info["r_frame_rate"].split("/")
        r_fps = float(num) / float(den)

        self.assertEqual(int(info["nb_frames"]), 2)
        self.assertAlmostEqual(r_fps, 1.0, delta=0.05)

    # ── 14: remux is skipped when the measured rate already matches ─────────

    def test_remux_skipped_when_rate_matches_declared(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")

        # cooldown_s ~= 1/declared_fps -> a 2-frame segment measures within
        # _FPS_REMUX_TOLERANCE of the declared fps, so _maybe_remux skips it.
        rec = self._make_recorder(cooldown_s=0.1)
        static_b64 = [_load_b64(f) for f in sorted((FIXTURES / "static_video").glob("droidcam_*.jpg"))[:2]]
        gripper_b64 = _load_b64(FIXTURES / "gripper_motion" / "droidcam_008.jpg")

        t = 1000.0
        rec.on_frame("droidcam", static_b64[0], t)
        t = round(t + 0.1, 3)
        rec.on_frame("droidcam", static_b64[1], t)
        t = round(t + 0.1, 3)
        trigger_ts = t
        rec.on_frame("droidcam", gripper_b64, trigger_ts)

        # +0.001 margin so `ts - last_motion_ts >= cooldown_s` holds despite
        # float rounding; actual_fps ~= 1/0.101 ~= 9.9, within 5% of 10.0.
        close_ts = round(trigger_ts + rec.cooldown_s + 0.001, 3)
        rec.on_frame("droidcam", gripper_b64, close_ts)

        seg = rec._cameras["droidcam"].recent_closed[-1]
        self.assertEqual(seg.frame_count, 2)

        cap = cv2.VideoCapture(seg.path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.assertAlmostEqual(fps, 10.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
