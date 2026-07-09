"""
Continuous motion-segment video recorder.

Taps frames already flowing through the camera frame caches
(`_PiFrameCache`/`_DroidCamFrameCache` in camera.py) and incrementally writes
motion-bounded mp4 segments to disk in real time. A manifest (JSONL) records
each closed segment's time range and path; `tag_range` lets callers attach
action metadata (tool name, change description) to whichever segment(s)
overlap a given time window.

Per-camera noise calibration
────────────────────────────
The first SEGMENT_CALIB_FRAMES stable frames received by each camera are used
to measure that camera's natural frame-to-frame pixel noise.  The motion
threshold is then set to mean_diff + SIGMA * std_diff above that noise floor,
where SIGMA is SEGMENT_CALIB_SIGMA_PI for the Pi Camera and
SEGMENT_CALIB_SIGMA for every other camera. This prevents the noisier Pi
Camera from triggering false-positive segments while the quieter DroidCam
keeps its baseline sensitivity.  Set SEGMENT_CALIB_ENABLED=0 to skip
calibration and use the fixed global threshold for all cameras.

Frames are only accumulated into the noise sample after SEGMENT_CALIB_WARMUP_S
seconds have passed since the camera's first frame (they still keep the
frame-diff reference fresh in the meantime). Both cameras' auto-exposure/
white-balance loops are still converging right after stream start, and
sampling during that window can measure an artificially quiet moment — see
the 2026-07-09 investigation, where both cameras calibrated in under two
seconds with a near-zero std_diff, then a genuine post-settling brightness
step opened a real-looking-but-empty ~0.5s segment on both cameras a few
seconds later. The warm-up delay lets the sensor get further into its
startup convergence before calibration starts measuring.

Cross-camera sync
─────────────────
Segment open/close decisions are driven by a single recorder-wide "last
motion" timestamp instead of a per-camera clock: any camera's own motion
refreshes it, and every camera (including the one that tripped) keeps
recording only while `now - last_motion < SEGMENT_COOLDOWN_S`. Previously
each camera tracked its own cross-trigger timestamp with its own, longer
cross-trigger window layered on top of the shorter per-camera cooldown —
two cameras watching the same physical action could each measure "last
motion" a frame or two apart, so one side's cooldown could lapse before the
other's cross-trigger refreshed it, closing one camera's segment early and
splitting an action that the other camera recorded as a single clip (see
2026-07-08 investigation). A single shared clock removes that race: both
streams open and close together, within about one frame interval, so VQA
always has a matched pair of clips.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from mcp_robot import config
from mcp_robot.vision import _CAPTURE_MOTION_THRESHOLD, _MOTION_PIXEL_THRESH, _MOTION_PIXEL_COUNT

log = logging.getLogger(__name__)
if config.SEGMENT_MOTION_LOG:
    log.setLevel(logging.DEBUG)


@dataclass
class _Segment:
    camera: str
    start_ts: float
    path: str
    writer: "object | None" = None
    width: int = 0
    height: int = 0
    frame_count: int = 0
    first_written_ts: float | None = None
    last_written_ts: float = 0.0
    end_ts: float | None = None
    closed: bool = False
    flushed: bool = False
    tool: str | None = None
    change_description: str | None = None
    sub_observation: str | None = None
    sub_action: str | None = None


@dataclass
class _CameraState:
    prev_gray: "np.ndarray | None" = None
    open_segment: "_Segment | None" = None
    recent_closed: "deque[_Segment]" = field(default_factory=deque)
    # Per-camera noise calibration
    calib_diffs: list = field(default_factory=list)
    calib_pixel_counts: list = field(default_factory=list)
    calib_done: bool = False
    calib_warmup_start_ts: "float | None" = None
    motion_threshold: float = _CAPTURE_MOTION_THRESHOLD
    motion_pixel_count: int = _MOTION_PIXEL_COUNT


@dataclass
class _MotionStats:
    motion: bool
    mean_diff: float
    n_changed: int
    tripped: "str | None"  # "mean" | "pixel_count" | None, whichever threshold fired


def _motion_stats(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    mean_threshold: float = _CAPTURE_MOTION_THRESHOLD,
    pixel_count: int = _MOTION_PIXEL_COUNT,
) -> _MotionStats:
    diff = np.abs(prev_gray.astype(np.float32) - cur_gray.astype(np.float32))
    mean_diff = float(diff.mean())
    n_changed = int(np.sum(diff > _MOTION_PIXEL_THRESH))
    motion = mean_diff > mean_threshold or n_changed > pixel_count
    tripped = ("mean" if mean_diff > mean_threshold else "pixel_count") if motion else None
    return _MotionStats(motion=motion, mean_diff=mean_diff, n_changed=n_changed, tripped=tripped)


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start <= b_end and a_end >= b_start


def _decode_frame(frame_b64: str) -> "np.ndarray | None":
    import base64
    import cv2

    buf = np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# Skip the remux when the measured rate is already this close to the declared
# rate — avoids a pointless ffmpeg call for segments recorded near their
# intended fps (e.g. the throttled per-action capture path).
_FPS_REMUX_TOLERANCE = 0.05


class SegmentRecorder:
    """Records per-camera motion-bounded mp4 segments and a JSONL manifest."""

    def __init__(
        self,
        segment_dir: str = config.SEGMENT_DIR,
        manifest_path: str = config.SEGMENT_MANIFEST,
        preroll_s: float = config.SEGMENT_PREROLL_S,
        cooldown_s: float = config.SEGMENT_COOLDOWN_S,
        fps_by_camera: dict | None = None,
        recent_ring: int = config.SEGMENT_RECENT_RING,
        calib_enabled: bool = config.SEGMENT_CALIB_ENABLED,
        calib_frames: int = config.SEGMENT_CALIB_FRAMES,
        calib_sigma: float = config.SEGMENT_CALIB_SIGMA,
        calib_sigma_by_camera: dict | None = None,
        calib_warmup_s: float = config.SEGMENT_CALIB_WARMUP_S,
    ) -> None:
        self.segment_dir = segment_dir
        self.manifest_path = manifest_path
        self.preroll_s = preroll_s
        self.cooldown_s = cooldown_s
        self.fps_by_camera = fps_by_camera or {
            "droidcam": config.SEGMENT_FPS_DROIDCAM,
            "pi_camera": config.SEGMENT_FPS_PI,
        }
        self.recent_ring = recent_ring
        self.calib_enabled = calib_enabled
        self.calib_frames = calib_frames
        self.calib_sigma = calib_sigma
        self.calib_sigma_by_camera = calib_sigma_by_camera or {"pi_camera": config.SEGMENT_CALIB_SIGMA_PI}
        self.calib_warmup_s = calib_warmup_s
        self._cameras: dict[str, _CameraState] = {}
        # Shared cross-camera clock (see module docstring): any camera's own
        # motion sets this, and every camera's open/close decision reads it —
        # not a per-camera timestamp — so all cameras agree on "how long ago
        # was the last motion anywhere" to the same value.
        self._last_motion_ts: float = 0.0
        # Which camera last drove _last_motion_ts — logged when a segment
        # opens so it's clear whether *this* camera tripped its own motion or
        # merely rode another camera's cooldown window (see on_frame).
        self._last_motion_camera: "str | None" = None
        self._lock = threading.Lock()
        os.makedirs(self.segment_dir, exist_ok=True)

    # ── calibration ─────────────────────────────────────────────────────────

    def _accumulate_calib(self, camera: str, state: _CameraState, gray: np.ndarray) -> None:
        """Accumulate one inter-frame diff sample; finalise when enough collected."""
        diff = np.abs(gray.astype(np.float32) - state.prev_gray.astype(np.float32))
        state.calib_diffs.append(float(diff.mean()))
        state.calib_pixel_counts.append(int(np.sum(diff > _MOTION_PIXEL_THRESH)))

        if len(state.calib_diffs) < self.calib_frames:
            return

        mean_d = float(np.mean(state.calib_diffs))
        std_d  = float(np.std(state.calib_diffs))
        mean_p = float(np.mean(state.calib_pixel_counts))
        std_p  = float(np.std(state.calib_pixel_counts))
        sigma  = self.calib_sigma_by_camera.get(camera, self.calib_sigma)

        # Raise the threshold above the noise floor; never drop below the global default.
        state.motion_threshold  = max(mean_d + sigma * std_d,  _CAPTURE_MOTION_THRESHOLD)
        state.motion_pixel_count = max(int(mean_p + sigma * std_p), _MOTION_PIXEL_COUNT)
        state.calib_done = True
        state.calib_diffs.clear()
        state.calib_pixel_counts.clear()

        log.info(
            "recorder: %s calibrated (σ=%.1f) — noise mean_diff=%.3f σ=%.3f → threshold=%.2f; "
            "mean_px=%.0f σ=%.0f → pixel_count=%d",
            camera, sigma, mean_d, std_d, state.motion_threshold,
            mean_p, std_p, state.motion_pixel_count,
        )

    # ── frame ingestion ─────────────────────────────────────────────────────

    def on_frame(self, camera: str, frame_b64: str, ts: float, cache=None) -> None:
        """Feed one frame from `camera`'s cache into the recorder."""
        import cv2
        import base64

        try:
            buf = np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                return
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as exc:
            log.debug("recorder.on_frame: failed to decode frame for %s: %s", camera, exc)
            return

        closed_seg = None
        with self._lock:
            state = self._cameras.setdefault(camera, _CameraState(recent_closed=deque(maxlen=self.recent_ring)))

            # ── calibration phase ───────────────────────────────────────────
            if self.calib_enabled and not state.calib_done:
                if state.calib_warmup_start_ts is None:
                    state.calib_warmup_start_ts = ts
                    log.debug(
                        "recorder: %s calibration warm-up started (ts=%.3f, warmup=%.1fs)",
                        camera, ts, self.calib_warmup_s,
                    )
                warmed_up = ts - state.calib_warmup_start_ts >= self.calib_warmup_s
                if warmed_up and state.prev_gray is not None:
                    self._accumulate_calib(camera, state, gray)
                state.prev_gray = gray
                return  # hold off recording until the noise floor is measured

            # ── motion detection ────────────────────────────────────────────
            own_stats = (
                None if state.prev_gray is None
                else _motion_stats(state.prev_gray, gray, state.motion_threshold, state.motion_pixel_count)
            )
            own_motion = own_stats.motion if own_stats is not None else False

            # Shared clock: own motion on *any* camera refreshes one
            # recorder-wide timestamp. Every camera's motion/no-motion
            # decision below reads that same value, so all cameras agree on
            # "how long ago was the last motion anywhere" instead of each
            # tracking its own, independently-timed view of it.
            if own_motion:
                self._last_motion_ts = ts
                self._last_motion_camera = camera
                log.debug(
                    "recorder: %s own_motion tripped=%s mean_diff=%.3f(thr=%.2f) changed_px=%d(thr=%d)",
                    camera, own_stats.tripped, own_stats.mean_diff, state.motion_threshold,
                    own_stats.n_changed, state.motion_pixel_count,
                )

            motion = ts - self._last_motion_ts < self.cooldown_s

            # ── segment logic ───────────────────────────────────────────────
            if state.open_segment is None:
                if motion:
                    seg = self._open_segment(camera, state, ts, cache)
                    self._write_frame(seg, frame_b64, ts)
                    state.open_segment = seg
                    # Which camera triggered this recording, and why — the
                    # only place this decision is made, so it's the only
                    # place that can log it (see 2026-07-09: calibration and
                    # remux lines were visible but nothing said what opened
                    # the segments in between).
                    if own_motion:
                        log.info(
                            "recorder: %s opening segment %s — own motion tripped=%s "
                            "mean_diff=%.3f(thr=%.2f) changed_px=%d(thr=%d)",
                            camera, os.path.basename(seg.path), own_stats.tripped, own_stats.mean_diff,
                            state.motion_threshold, own_stats.n_changed, state.motion_pixel_count,
                        )
                    else:
                        log.info(
                            "recorder: %s opening segment %s — cross-camera trigger from %s (%.3fs ago)",
                            camera, os.path.basename(seg.path), self._last_motion_camera,
                            ts - self._last_motion_ts,
                        )
            else:
                seg = state.open_segment
                self._write_frame(seg, frame_b64, ts)
                if not motion:
                    self._close_segment(camera, state)
                    closed_seg = seg

            state.prev_gray = gray

        if closed_seg is not None:
            self._maybe_remux(closed_seg)

    # ── segment lifecycle ───────────────────────────────────────────────────

    def _open_segment(self, camera: str, state: _CameraState, trigger_ts: float, cache) -> _Segment:
        path = os.path.join(self.segment_dir, f"{camera}_{trigger_ts:.3f}.mp4")
        seg = _Segment(camera=camera, start_ts=trigger_ts, path=path)

        if cache is not None:
            try:
                preroll = cache.clip_since(trigger_ts - self.preroll_s, max_fps=self.fps_by_camera.get(camera, 15.0))
            except Exception as exc:
                log.debug("recorder._open_segment: clip_since failed for %s: %s", camera, exc)
                preroll = None
            if preroll:
                for f in preroll:
                    if f["ts"] < trigger_ts:
                        self._write_frame(seg, f["frame"], f["ts"])

        return seg

    def _write_frame(self, seg: _Segment, frame_b64: str, ts: float) -> None:
        # Pre-existing behavior for the continuous live-stream path (unrelated
        # to log_thought, see below): an occasional corrupt frame from the
        # camera must not crash on_frame()'s hot path.
        try:
            img = _decode_frame(frame_b64)
        except Exception as exc:
            log.debug("recorder._write_frame: decode failed for %s: %s", seg.path, exc)
            return
        if img is None:
            return
        self._write_decoded_frame(seg, img, ts)

    def _write_decoded_frame(self, seg: _Segment, img: np.ndarray, ts: float) -> None:
        import cv2

        if seg.writer is None:
            h, w = img.shape[:2]
            seg.width, seg.height = w, h
            fourcc = cv2.VideoWriter_fourcc(*config.SEGMENT_FOURCC)
            fps = self.fps_by_camera.get(seg.camera, 15.0)
            seg.writer = cv2.VideoWriter(seg.path, fourcc, fps, (w, h))
            if not seg.writer.isOpened():
                log.warning("recorder: failed to open VideoWriter for %s", seg.path)
                seg.writer = None
                return

        if img.shape[1] != seg.width or img.shape[0] != seg.height:
            img = cv2.resize(img, (seg.width, seg.height))

        seg.writer.write(img)
        if seg.frame_count == 0:
            seg.first_written_ts = ts
        seg.frame_count += 1
        seg.last_written_ts = ts

    def _close_segment(self, camera: str, state: _CameraState) -> None:
        seg = state.open_segment
        if seg is None:
            return
        if seg.writer is not None:
            seg.writer.release()
            seg.writer = None
        seg.end_ts = seg.last_written_ts
        seg.closed = True
        duration = (seg.last_written_ts - seg.first_written_ts) if seg.first_written_ts is not None else 0.0
        log.info(
            "recorder: %s closing segment %s (%d frames over %.2fs)",
            camera, os.path.basename(seg.path), seg.frame_count, duration,
        )
        self._append_manifest(seg)
        state.recent_closed.append(seg)
        state.open_segment = None

    def _maybe_remux(self, seg: _Segment) -> None:
        """Rewrite seg's container timestamps so playback fps matches the
        measured arrival rate, if it drifted from the declared fps used when
        opening the VideoWriter (continuous streams run at the camera's
        native rate, not the declared SEGMENT_FPS_*). Not called while holding
        self._lock — runs after the lock is released so it never blocks the
        other camera's on_frame()."""
        if seg.frame_count <= 1 or seg.first_written_ts is None:
            return
        duration = seg.last_written_ts - seg.first_written_ts
        if duration <= 0:
            return
        actual_fps = (seg.frame_count - 1) / duration
        declared_fps = self.fps_by_camera.get(seg.camera, 15.0)
        if declared_fps <= 0 or actual_fps <= 0:
            return
        if abs(actual_fps - declared_fps) / declared_fps < _FPS_REMUX_TOLERANCE:
            return

        itsscale = declared_fps / actual_fps
        tmp_path = seg.path + ".remux.mp4"
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-itsscale", f"{itsscale:.6f}", "-i", seg.path,
                 "-c", "copy", "-fflags", "+genpts", tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                os.replace(tmp_path, seg.path)
                log.info("recorder: remuxed %s to %.2f fps (declared %.2f, %d frames over %.2fs)",
                         seg.path, actual_fps, declared_fps, seg.frame_count, duration)
            else:
                log.warning("recorder: remux failed for %s: %s", seg.path, proc.stderr[-500:])
        except Exception as exc:
            log.warning("recorder: remux error for %s: %s", seg.path, exc)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ── thought segments ─────────────────────────────────────────────────────

    def log_thought(
        self,
        sub_observation: str,
        sub_action: str,
        frames_by_camera: dict[str, str],
        duration_s: float = config.THOUGHT_SEGMENT_DURATION_S,
        now: float | None = None,
    ) -> bool:
        """Synthesize a frozen-frame segment per camera in `frames_by_camera`
        (camera -> base64 JPEG), each spanning a virtual [now, now+duration_s]
        window with tool="thought" — for a diagnostic moment that has no real
        motion to tag. `now` defaults to time.time(); callers (tests) may
        pass an explicit value to align with a synthetic clock. Segments are
        permanently exempt from tag_range() (see
        the `seg.tool != "thought"` guards there) since a synthetic end_ts can
        still be in the future when the next real action's tag_range call
        lands, which would otherwise let it overwrite this narration.

        Each camera's segment is built and written entirely outside
        self._lock: on_frame() is called synchronously, inline, from the same
        thread that reads each live camera stream (see camera.py's
        _PiFrameCache.put/_DroidCamFrameCache.put), so holding the lock for
        the ~45-90 frames written here would stall those threads and drop
        real, physical frames arriving meanwhile. The lock is only taken
        briefly at the end, per camera, to pair the manifest append with the
        recent_closed update atomically.

        Returns True if at least one camera produced a segment.
        """
        start_ts = time.time() if now is None else now
        end_ts = start_ts + duration_s
        any_ok = False
        for camera, frame_b64 in frames_by_camera.items():
            img = _decode_frame(frame_b64)
            if img is None:
                continue
            fps = self.fps_by_camera.get(camera, 15.0)
            path = os.path.join(self.segment_dir, f"{camera}_{start_ts:.3f}_thought.mp4")
            seg = _Segment(
                camera=camera, start_ts=start_ts, path=path, tool="thought",
                sub_observation=sub_observation or None, sub_action=sub_action or None,
            )
            n_frames = max(1, int(round(duration_s * fps)))
            for i in range(n_frames):
                self._write_decoded_frame(seg, img, start_ts + i / fps)
            if seg.writer is not None:
                seg.writer.release()
                seg.writer = None
            if seg.frame_count == 0:
                continue  # VideoWriter never opened; nothing to record
            seg.end_ts = end_ts
            seg.closed = True
            with self._lock:
                self._append_manifest(seg)
                state = self._cameras.setdefault(
                    camera, _CameraState(recent_closed=deque(maxlen=self.recent_ring))
                )
                state.recent_closed.append(seg)
            any_ok = True
        return any_ok

    # ── manifest I/O (always called while holding self._lock) ──────────────

    def _segment_record(self, seg: _Segment) -> dict:
        return {
            "camera": seg.camera,
            "start_ts": round(seg.start_ts, 3),
            "end_ts": round(seg.end_ts, 3) if seg.end_ts is not None else None,
            "path": seg.path,
            "frame_count": seg.frame_count,
            "tool": seg.tool,
            "change_description": seg.change_description,
            "sub_observation": seg.sub_observation,
            "sub_action": seg.sub_action,
        }

    def _append_manifest(self, seg: _Segment) -> None:
        try:
            with open(self.manifest_path, "a") as fh:
                fh.write(json.dumps(self._segment_record(seg)) + "\n")
            seg.flushed = True
        except Exception as exc:
            log.warning("recorder: failed to append manifest entry for %s: %s", seg.path, exc)

    def _rewrite_manifest_line(self, seg: _Segment) -> None:
        try:
            with open(self.manifest_path, "r") as fh:
                lines = fh.readlines()
        except FileNotFoundError:
            self._append_manifest(seg)
            return
        except Exception as exc:
            log.warning("recorder: failed to read manifest for rewrite: %s", exc)
            return

        new_record = self._segment_record(seg)
        found = False
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("camera") == seg.camera and abs(rec.get("start_ts", -1e18) - new_record["start_ts"]) < 1e-3:
                lines[i] = json.dumps(new_record) + "\n"
                found = True
                break

        if not found:
            lines.append(json.dumps(new_record) + "\n")

        try:
            with open(self.manifest_path, "w") as fh:
                fh.writelines(lines)
        except Exception as exc:
            log.warning("recorder: failed to write manifest: %s", exc)

    def _tag_manifest_fallback(self, camera: str, t_start: float, t_end: float, meta: dict) -> bool:
        try:
            with open(self.manifest_path, "r") as fh:
                lines = fh.readlines()
        except FileNotFoundError:
            return False
        except Exception as exc:
            log.warning("recorder: failed to read manifest for fallback tag: %s", exc)
            return False

        tagged = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except Exception:
                continue
            if rec.get("camera") != camera or rec.get("end_ts") is None or rec.get("tool") == "thought":
                continue
            if _overlaps(rec["start_ts"], rec["end_ts"], t_start, t_end):
                rec.update(meta)
                lines[i] = json.dumps(rec) + "\n"
                tagged = True

        if tagged:
            try:
                with open(self.manifest_path, "w") as fh:
                    fh.writelines(lines)
            except Exception as exc:
                log.warning("recorder: failed to write manifest for fallback tag: %s", exc)
                return False

        return tagged

    # ── tagging ──────────────────────────────────────────────────────────────

    def tag_range(self, camera: str, t_start: float, t_end: float, meta: dict) -> bool:
        """Attach `meta` (e.g. {"tool": ..., "change_description": ...}) to every
        segment for `camera` that overlaps [t_start, t_end]. Returns False if no
        segment overlaps (e.g. a no-motion action)."""
        with self._lock:
            tagged_any = False

            state = self._cameras.get(camera)
            if state is not None:
                seg = state.open_segment
                if (seg is not None and seg.tool != "thought"
                        and _overlaps(seg.start_ts, time.time(), t_start, t_end)):
                    seg.tool = meta.get("tool", seg.tool)
                    seg.change_description = meta.get("change_description", seg.change_description)
                    seg.sub_observation = meta.get("sub_observation", seg.sub_observation)
                    seg.sub_action = meta.get("sub_action", seg.sub_action)
                    tagged_any = True

                # tool == "thought" segments (log_thought) are permanently
                # exempt: their end_ts is a synthetic now+duration_s, which can
                # still be in the future when a real action's window lands, so
                # without this guard a real tag_range() call could silently
                # overwrite a diagnostic narration with motor-tool metadata.
                for seg in state.recent_closed:
                    if (seg.tool != "thought" and seg.end_ts is not None
                            and _overlaps(seg.start_ts, seg.end_ts, t_start, t_end)):
                        seg.tool = meta.get("tool", seg.tool)
                        seg.change_description = meta.get("change_description", seg.change_description)
                        seg.sub_observation = meta.get("sub_observation", seg.sub_observation)
                        seg.sub_action = meta.get("sub_action", seg.sub_action)
                        self._rewrite_manifest_line(seg)
                        tagged_any = True

            if not tagged_any:
                tagged_any = self._tag_manifest_fallback(camera, t_start, t_end, meta)

            return tagged_any

    # ── shutdown ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        closed_segs = []
        with self._lock:
            for camera, state in self._cameras.items():
                if state.open_segment is not None:
                    seg = state.open_segment
                    self._close_segment(camera, state)
                    closed_segs.append(seg)
        for seg in closed_segs:
            self._maybe_remux(seg)


_recorder: SegmentRecorder | None = None


def get_recorder() -> SegmentRecorder:
    global _recorder
    if _recorder is None:
        _recorder = SegmentRecorder()
    return _recorder
