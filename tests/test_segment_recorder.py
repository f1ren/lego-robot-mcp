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
import time
import unittest

import cv2

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _load_b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _solid_b64(width: int, height: int, value: int) -> str:
    """A solid-color width×height JPEG, base64-encoded. Two different
    `value`s produce a guaranteed-large frame diff regardless of the
    recorder's calibrated motion threshold — used to drive segment open/close
    deterministically without depending on real footage's motion timing."""
    import numpy as np
    img = np.full((height, width, 3), value, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


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
            calib_enabled=False,  # skip noise calibration in unit tests
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

        # +0.001 margin so `ts - _last_motion_ts >= cooldown_s` holds despite
        # float rounding; actual_fps ~= 1/0.101 ~= 9.9, within 5% of 10.0.
        close_ts = round(trigger_ts + rec.cooldown_s + 0.001, 3)
        rec.on_frame("droidcam", gripper_b64, close_ts)

        seg = rec._cameras["droidcam"].recent_closed[-1]
        self.assertEqual(seg.frame_count, 2)

        cap = cv2.VideoCapture(seg.path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.assertAlmostEqual(fps, 10.0, delta=0.01)


    # ── 15: noise calibration raises threshold and blocks early segments ──────

    def test_calibration_raises_threshold_and_delays_recording(self):
        """Calibration phase should consume calib_frames before recording starts,
        and the measured threshold must be >= the global default."""
        import numpy as np
        import cv2

        rec = self._make_recorder(calib_enabled=True, calib_frames=4, calib_sigma=3.0)

        # Build synthetic frames with tiny pixel noise (mean diff ~1, well below 2.5).
        rng = np.random.default_rng(42)
        base = np.full((100, 100, 3), 128, dtype=np.uint8)

        def _noisy_b64():
            noise = rng.integers(-2, 3, base.shape, dtype=np.int16)
            frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            _, buf = cv2.imencode(".jpg", frame)
            return base64.b64encode(buf.tobytes()).decode()

        t = 1000.0
        # calib_frames=4 diffs require calib_frames+1 total frames (first just seeds prev_gray).
        for _ in range(5):
            rec.on_frame("droidcam", _noisy_b64(), t)
            t = round(t + 0.1, 3)

        state = rec._cameras["droidcam"]
        self.assertTrue(state.calib_done, "calibration should be done after calib_frames+1 frames")
        self.assertIsNone(state.open_segment, "no segment should open during calibration")
        # Threshold must be at or above the global default.
        from mcp_robot.vision import _CAPTURE_MOTION_THRESHOLD
        self.assertGreaterEqual(state.motion_threshold, _CAPTURE_MOTION_THRESHOLD)

    # ── 16: cross-trigger opens a segment on the second camera ───────────────

    def test_cross_trigger_opens_segment_on_second_camera(self):
        """Motion on droidcam should open a segment on pi_camera via the
        shared motion clock even if pi_camera has no own motion."""
        rec = self._make_recorder(
            fps_by_camera={"droidcam": 10.0, "pi_camera": 10.0},
        )

        static_b64 = [_load_b64(f) for f in sorted((FIXTURES / "static_video").glob("droidcam_*.jpg"))[:2]]
        gripper_b64 = _load_b64(FIXTURES / "gripper_motion" / "droidcam_008.jpg")

        t = 1000.0
        # Seed pi_camera with one static frame (no motion, no segment).
        rec.on_frame("pi_camera", static_b64[0], t)
        t = round(t + 0.05, 3)

        # droidcam: two static + one motion frame → segment opens.
        rec.on_frame("droidcam", static_b64[0], t); t = round(t + 0.1, 3)
        rec.on_frame("droidcam", static_b64[1], t); t = round(t + 0.1, 3)
        rec.on_frame("droidcam", gripper_b64,   t)  # triggers motion

        # pi_camera: feed one static frame shortly after — cross-trigger should fire.
        t = round(t + 0.05, 3)
        rec.on_frame("pi_camera", static_b64[1], t)

        pi_state = rec._cameras.get("pi_camera")
        self.assertIsNotNone(pi_state, "pi_camera state should exist")
        self.assertIsNotNone(
            pi_state.open_segment,
            "pi_camera should have an open segment due to cross-trigger from droidcam motion",
        )

    # ── 17: shared clock keeps all cameras' segments in sync ────────────────

    def test_shared_clock_closes_all_cameras_together(self):
        """Regression test for the 2026-07-08 segment misalignment: a camera
        whose own motion has already stopped must stay open as long as *any*
        camera keeps tripping its own detector, and once everything goes
        quiet, every camera closes off the same shared instant instead of
        racing on independent per-camera cooldowns (which previously let one
        camera's segment end several seconds before its sibling's for the
        same action)."""
        rec = self._make_recorder(fps_by_camera={"droidcam": 10.0, "pi_camera": 10.0})

        lo, hi = _solid_b64(64, 64, 40), _solid_b64(64, 64, 220)

        t = 1000.0
        # Both cameras trip their own motion at ~the same time and open.
        rec.on_frame("droidcam", lo, t)
        rec.on_frame("pi_camera", lo, t)
        t = round(t + 0.1, 3)
        rec.on_frame("droidcam", hi, t)
        rec.on_frame("pi_camera", hi, t)

        droid_state = rec._cameras["droidcam"]
        pi_state = rec._cameras["pi_camera"]
        self.assertIsNotNone(droid_state.open_segment)
        self.assertIsNotNone(pi_state.open_segment)

        # pi_camera goes still (repeats the same frame -> no own motion from
        # here on); droidcam keeps alternating -> its own motion keeps
        # tripping and should keep *both* cameras' segments open.
        last_motion_ts = t
        for droid_frame in (lo, hi, lo):
            t = round(t + 0.05, 3)
            rec.on_frame("pi_camera", hi, t)          # identical -> no own motion
            rec.on_frame("droidcam", droid_frame, t)  # alternates -> own motion trips
            last_motion_ts = t

        self.assertIsNotNone(
            pi_state.open_segment,
            "pi_camera must stay open: droidcam's own motion refreshes the shared clock",
        )

        # Now both go quiet. A single frame past cooldown on each should
        # close both, off the *same* shared last-motion instant.
        close_ts = round(last_motion_ts + rec.cooldown_s + 0.001, 3)
        rec.on_frame("droidcam", lo, close_ts)   # identical to last droidcam frame -> no own motion
        rec.on_frame("pi_camera", hi, close_ts)  # identical to last pi_camera frame -> no own motion

        self.assertIsNone(droid_state.open_segment)
        self.assertIsNone(pi_state.open_segment)
        self.assertEqual(
            droid_state.recent_closed[-1].end_ts,
            pi_state.recent_closed[-1].end_ts,
            "both cameras must close off the same shared last-motion instant",
        )

    # ── 18: compile_merged_video tiles both cameras + subtitle ──────────────

    def test_compile_merged_video(self):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg/ffprobe not available")

        rec = self._make_recorder(fps_by_camera={"droidcam": 10.0, "pi_camera": 10.0})

        # Synthetic solid-color frames (not real fixtures): droidcam portrait
        # 480x640 (like a rotated DroidCam feed), pi_camera landscape 640x480
        # — guarantees deterministic motion (huge frame diff) and exercises
        # aspect-preserving scale/pad for both orientations.
        droid_w, droid_h = 480, 640
        pi_w, pi_h = 640, 480
        droid_lo, droid_hi = _solid_b64(droid_w, droid_h, 40), _solid_b64(droid_w, droid_h, 220)
        pi_lo, pi_hi = _solid_b64(pi_w, pi_h, 40), _solid_b64(pi_w, pi_h, 220)

        t0 = 1000.0
        t = t0
        rec.on_frame("droidcam", droid_lo, t); t = round(t + 0.1, 3)
        rec.on_frame("droidcam", droid_hi, t); t = round(t + 0.1, 3)
        rec.on_frame("droidcam", droid_lo, t); t = round(t + 0.1, 3)
        droid_close = round(t + rec.cooldown_s, 3)
        rec.on_frame("droidcam", droid_lo, droid_close)

        t = round(t0 + 0.05, 3)  # slightly offset from droidcam, within cross-trigger window
        rec.on_frame("pi_camera", pi_lo, t); t = round(t + 0.1, 3)
        rec.on_frame("pi_camera", pi_hi, t); t = round(t + 0.1, 3)
        rec.on_frame("pi_camera", pi_lo, t); t = round(t + 0.1, 3)
        pi_close = round(t + rec.cooldown_s, 3)
        rec.on_frame("pi_camera", pi_lo, pi_close)

        meta = {"tool": "drive", "sub_observation": "Path is clear", "sub_action": "Driving backward"}
        self.assertTrue(rec.tag_range("droidcam", t0, droid_close, meta))
        self.assertTrue(rec.tag_range("pi_camera", t0, pi_close, meta))

        from mcp_robot.video_compiler import compile_merged_video
        result = compile_merged_video(since=str(t0 - 1), manifest_path=rec.manifest_path, out_dir=self.tmpdir)
        self.assertTrue(result.ok, result.error)
        self.assertIsNotNone(result.video_path)
        self.assertTrue(os.path.exists(result.video_path))
        self.assertEqual(result.segment_count, 2)
        self.assertGreater(result.total_duration_s, 0)

        # Expected geometry: mirrors compile_merged_video's own math.
        from mcp_robot import config
        from mcp_robot.video_compiler import _even

        cell_h = _even(config.MERGE_CELL_HEIGHT)
        cell_w = _even(cell_h * pi_w / pi_h)
        right_h = cell_h * 2
        right_w = _even(right_h * droid_w / droid_h)

        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", result.video_path],
            capture_output=True, text=True,
        )
        w_str, h_str = out.stdout.strip().split("x")
        self.assertEqual(int(h_str), right_h)
        self.assertEqual(int(w_str), cell_w + right_w)

        # Content check: the top-left (subtitle-on-black) cell must be much
        # darker on average than the bottom-left (real camera footage) cell —
        # catches an accidental pane swap that dimension checks alone would miss.
        cap = cv2.VideoCapture(result.video_path)
        ok, frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        top_left = frame[0:cell_h, 0:cell_w]
        bottom_left = frame[cell_h:2 * cell_h, 0:cell_w]
        self.assertLess(top_left.mean(), bottom_left.mean())


    # ── 19: log_thought produces well-formed manifest entries per camera ────

    def test_log_thought_produces_manifest_entries(self):
        rec = self._make_recorder(fps_by_camera={"droidcam": 10.0, "pi_camera": 10.0})
        frame = _solid_b64(64, 64, 100)

        ok = rec.log_thought("No cup found", "Consulting PDDL domain",
                              {"droidcam": frame, "pi_camera": frame}, duration_s=0.5)
        self.assertTrue(ok)

        records = _read_manifest(rec.manifest_path)
        self.assertEqual(len(records), 2)
        by_camera = {r["camera"]: r for r in records}
        self.assertAlmostEqual(by_camera["droidcam"]["start_ts"], by_camera["pi_camera"]["start_ts"], places=3)
        for camera, row in by_camera.items():
            self.assertEqual(row["tool"], "thought")
            self.assertEqual(row["sub_observation"], "No cup found")
            self.assertEqual(row["sub_action"], "Consulting PDDL domain")
            self.assertIsNone(row["change_description"])
            expected_frames = round(0.5 * rec.fps_by_camera[camera])
            self.assertEqual(row["frame_count"], expected_frames)
            self.assertTrue(os.path.isfile(row["path"]))
            self.assertEqual(_frame_count(row["path"]), expected_frames)

    # ── 20: log_thought returns promptly, not blocked for duration_s ────────

    def test_log_thought_returns_promptly(self):
        rec = self._make_recorder(fps_by_camera={"droidcam": 30.0, "pi_camera": 15.0})
        frame = _solid_b64(64, 64, 100)

        t_wall = time.time()
        ok = rec.log_thought("Observation", "Action",
                              {"droidcam": frame, "pi_camera": frame}, duration_s=3.0)
        elapsed = time.time() - t_wall

        self.assertTrue(ok)
        self.assertLess(elapsed, 1.0, "log_thought must synthesize frames, not sleep for duration_s")

    # ── 21: tag_range never overwrites a thought segment ────────────────────

    def test_tag_range_does_not_overwrite_thought_segment(self):
        rec = self._make_recorder()
        frame = _solid_b64(64, 64, 100)
        rec.log_thought("No cup found", "Consulting PDDL domain", {"droidcam": frame}, duration_s=0.5)

        records_before = _read_manifest(rec.manifest_path)
        self.assertEqual(len(records_before), 1)
        seg_start, seg_end = records_before[0]["start_ts"], records_before[0]["end_ts"]

        ok = rec.tag_range("droidcam", seg_start, seg_end,
                            {"tool": "drive", "sub_observation": "x", "sub_action": "Driving forward"})
        self.assertFalse(ok, "a thought segment must never be reported as tagged")

        records_after = _read_manifest(rec.manifest_path)
        self.assertEqual(records_after[0]["tool"], "thought")
        self.assertEqual(records_after[0]["sub_observation"], "No cup found")
        self.assertEqual(records_after[0]["sub_action"], "Consulting PDDL domain")

    # ── 22: tag_range still tags a real segment sitting next to a thought ───

    def test_tag_range_still_tags_real_segment_next_to_thought(self):
        rec = self._make_recorder()
        _, seg1 = self._build_closed_gripper_segment(rec, t0=1000.0)

        frame = _solid_b64(64, 64, 100)
        rec.log_thought("Note", "Thinking", {"droidcam": frame}, duration_s=0.5, now=seg1.start_ts)

        state = rec._cameras["droidcam"]
        self.assertEqual(len(state.recent_closed), 2, "real segment + thought segment")

        ok = rec.tag_range("droidcam", seg1.start_ts, seg1.end_ts,
                            {"tool": "control_gripper", "change_description": "gripper opened"})
        self.assertTrue(ok)

        records = _read_manifest(rec.manifest_path)
        real_rec = next(r for r in records if r["path"] == seg1.path)
        thought_rec = next(r for r in records if r["tool"] == "thought")
        self.assertEqual(real_rec["tool"], "control_gripper")
        self.assertEqual(real_rec["change_description"], "gripper opened")
        self.assertEqual(thought_rec["sub_observation"], "Note")

    # ── 23: two distinct log_thought calls produce two distinct scenes ──────

    def test_two_log_thought_calls_produce_distinct_scenes(self):
        rec = self._make_recorder(fps_by_camera={"droidcam": 10.0, "pi_camera": 10.0})
        frame = _solid_b64(64, 64, 100)

        t0 = 1000.0
        rec.log_thought("No cup found", "Consulting PDDL domain",
                         {"droidcam": frame, "pi_camera": frame}, duration_s=0.5, now=t0)
        # 1.0s later: within the old (buggy) gap_s(0.5) + duration_s(0.5) window
        # of the first call's end, but a distinct log_thought() call.
        t1 = t0 + 1.0
        rec.log_thought("Still stuck", "Re-planning",
                         {"droidcam": frame, "pi_camera": frame}, duration_s=0.5, now=t1)

        records = _read_manifest(rec.manifest_path)
        self.assertEqual(len(records), 4)  # 2 cameras x 2 calls
        records.sort(key=lambda r: r["start_ts"])

        from mcp_robot import config
        from mcp_robot.video_compiler import _group_into_scenes, _scene_subtitle
        scenes = _group_into_scenes(records, config.MERGE_SCENE_GAP_S)

        self.assertEqual(len(scenes), 2, "two distinct log_thought() calls must not merge into one scene")
        self.assertEqual(len(scenes[0]), 2, "both cameras from the same call must still share one scene")
        self.assertEqual(len(scenes[1]), 2)
        subs = [_scene_subtitle(s) for s in scenes]
        self.assertEqual(subs[0], ("No cup found", "Consulting PDDL domain"))
        self.assertEqual(subs[1], ("Still stuck", "Re-planning"))


if __name__ == "__main__":
    unittest.main()
