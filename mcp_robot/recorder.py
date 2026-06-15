"""
Continuous motion-segment video recorder.

Taps frames already flowing through the camera frame caches
(`_PiFrameCache`/`_DroidCamFrameCache` in camera.py) and incrementally writes
motion-bounded mp4 segments to disk in real time. A manifest (JSONL) records
each closed segment's time range and path; `tag_range` lets callers attach
action metadata (tool name, change description) to whichever segment(s)
overlap a given time window.
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


@dataclass
class _Segment:
    camera: str
    start_ts: float
    path: str
    writer: "object | None" = None
    width: int = 0
    height: int = 0
    frame_count: int = 0
    last_motion_ts: float = 0.0
    first_written_ts: float | None = None
    last_written_ts: float = 0.0
    end_ts: float | None = None
    closed: bool = False
    flushed: bool = False
    tool: str | None = None
    change_description: str | None = None


@dataclass
class _CameraState:
    prev_gray: "np.ndarray | None" = None
    open_segment: "_Segment | None" = None
    recent_closed: "deque[_Segment]" = field(default_factory=deque)


def _is_motion(prev_gray: np.ndarray, cur_gray: np.ndarray) -> bool:
    diff = np.abs(prev_gray.astype(np.float32) - cur_gray.astype(np.float32))
    if float(diff.mean()) > _CAPTURE_MOTION_THRESHOLD:
        return True
    return int(np.sum(diff > _MOTION_PIXEL_THRESH)) > _MOTION_PIXEL_COUNT


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start <= b_end and a_end >= b_start


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
        self._cameras: dict[str, _CameraState] = {}
        self._lock = threading.Lock()
        os.makedirs(self.segment_dir, exist_ok=True)

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

            motion = False if state.prev_gray is None else _is_motion(state.prev_gray, gray)

            if state.open_segment is None:
                if motion:
                    seg = self._open_segment(camera, state, ts, cache)
                    self._write_frame(seg, frame_b64, ts)
                    seg.last_motion_ts = ts
                    state.open_segment = seg
            else:
                seg = state.open_segment
                self._write_frame(seg, frame_b64, ts)
                if motion:
                    seg.last_motion_ts = ts
                elif ts - seg.last_motion_ts >= self.cooldown_s:
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
        import cv2
        import base64

        try:
            buf = np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception as exc:
            log.debug("recorder._write_frame: decode failed for %s: %s", seg.path, exc)
            return
        if img is None:
            return

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
            if rec.get("camera") != camera or rec.get("end_ts") is None:
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
                if seg is not None and _overlaps(seg.start_ts, time.time(), t_start, t_end):
                    seg.tool = meta.get("tool", seg.tool)
                    seg.change_description = meta.get("change_description", seg.change_description)
                    tagged_any = True

                for seg in state.recent_closed:
                    if seg.end_ts is not None and _overlaps(seg.start_ts, seg.end_ts, t_start, t_end):
                        seg.tool = meta.get("tool", seg.tool)
                        seg.change_description = meta.get("change_description", seg.change_description)
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
