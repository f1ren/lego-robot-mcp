"""
Scene text: notice new text (signs, labels, warnings) in the live camera
streams and raise it to the coordinator (the MCP caller) as a visual signal.

    camera frame ──► SceneTextWatcher.on_frame        (camera thread — cheap)
                       every SCENE_TEXT_SAMPLE_INTERVAL_S: 1/4-scale grayscale
                       decode; fraction of pixels changed since the last
                       sample → moving or still
                     still for SCENE_TEXT_STABLE_S, and the view differs from
                     the last one read (SCENE_TEXT_CHANGE_FRAC) → worker
                       ▼
                     worker thread: clear? (brightness, sharpness) → OCR
                     (RapidOCR, ~0.2s on CPU) → drop low-confidence and
                     edge-truncated lines → any word not seen before?
                       ▼
                     visual signal: frames → SCENE_TEXT_DIR,
                                    one JSON line → SIGNALS_OUTBOX
                       ▼
                     SignalBox — the coordinator hears about it twice over:
                       - background: tailing SIGNALS_OUTBOX (Claude Code's
                         Monitor tool) delivers it between tool calls;
                       - in-band: the next tool result carries it as
                         "visual_signals", and until one has, the robot is
                         paused — multi-step motions stop at their next step
                         and no motor command starts.

Deciding whether a signal matters, and replanning if it does, is the
coordinator's job, not this server's: it knows the task, and it talks to the
planner. The pause exists because the coordinator can't act while a long
tool call (navigate_to) is still running, and a signal read while the robot
is stopped must not be missed if the background tail isn't running.

Only still views are read — text in a moving frame is blurred and OCR
guesses on it — and only views that changed since the last read, so a
robot parked in front of a sign costs one OCR call, not one per frame.
"""
from __future__ import annotations

import base64
import difflib
import json
import logging
import os
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from mcp_robot import config

log = logging.getLogger(__name__)

CAMERA_DESCRIPTIONS = {
    "pi_camera": "front camera on the robot (robot's-eye view, facing where the gripper points)",
    "simpleipcamera": "external camera overlooking the robot and its surroundings",
}


# ── OCR ──────────────────────────────────────────────────────────────────────

@dataclass
class TextLine:
    text: str
    score: float
    box: list[list[float]]  # 4 corners, pixel coords

    def to_dict(self) -> dict:
        return {"text": self.text, "score": round(self.score, 3), "box": [[round(x), round(y)] for x, y in self.box]}


_ocr = None
_ocr_lock = threading.Lock()
_ocr_failed = False


def _get_ocr():
    """Lazily-built RapidOCR engine, or None if it isn't installed."""
    global _ocr, _ocr_failed
    with _ocr_lock:
        if _ocr is None and not _ocr_failed:
            try:
                from rapidocr_onnxruntime import RapidOCR
                _ocr = RapidOCR(intra_op_num_threads=config.SCENE_TEXT_OCR_THREADS)
            except Exception as exc:  # ImportError, or a broken onnxruntime
                _ocr_failed = True
                log.warning("scene_text: OCR unavailable (%s) — scene text is off. "
                            "pip install rapidocr_onnxruntime", exc)
        return _ocr


def ocr_available() -> bool:
    return _get_ocr() is not None


def read_text(
    bgr: np.ndarray,
    min_score: float = config.SCENE_TEXT_MIN_SCORE,
    border_px: int = config.SCENE_TEXT_BORDER_PX,
) -> list[TextLine]:
    """Text lines in a BGR frame, minus low-confidence ones (icons read as
    "#"), ones with fewer than 2 letters, and ones touching the frame edge —
    text cut off by the edge reads as a partial word. Letters, not digits:
    the external camera has caught a clock-like "15:4627" whose digits would
    be new on every read (seen on 2026-09-30's recordings)."""
    ocr = _get_ocr()
    if ocr is None:
        return []
    result, _ = ocr(bgr)
    h, w = bgr.shape[:2]
    lines = []
    for box, text, score in result or []:
        text, score = str(text).strip(), float(score)
        if score < min_score or sum(ch.isalpha() for ch in text) < 2:
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        if min(xs) <= border_px or min(ys) <= border_px or max(xs) >= w - 1 - border_px or max(ys) >= h - 1 - border_px:
            continue
        lines.append(TextLine(text, score, [[float(x), float(y)] for x, y in box]))
    return lines


# ── novelty ──────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) >= 2]


class SeenTextRegistry:
    """Words read so far this session. Novelty is decided per word, not per
    line, because OCR splits and merges the same sign's lines differently
    from frame to frame ("Glass Plastic" + "Paper" vs "Glass" + "Plastic
    Paper"); fuzzy matching absorbs misreads ("Plastlc")."""

    def __init__(self, match_ratio: float = config.SCENE_TEXT_MATCH_RATIO) -> None:
        self.match_ratio = match_ratio
        self._words: set[str] = set()
        self._lock = threading.Lock()

    def _known(self, word: str) -> bool:
        return word in self._words or any(
            difflib.SequenceMatcher(None, word, seen).ratio() >= self.match_ratio for seen in self._words
        )

    def novel(self, lines: list[TextLine]) -> list[TextLine]:
        """Lines containing at least one word not seen before."""
        with self._lock:
            return [line for line in lines if any(not self._known(w) for w in words(line.text))]

    def remember(self, lines: list[TextLine]) -> None:
        with self._lock:
            for line in lines:
                self._words.update(words(line.text))

    def clear(self) -> None:
        with self._lock:
            self._words.clear()

    def snapshot(self) -> list[str]:
        with self._lock:
            return sorted(self._words)


# ── frame tests ──────────────────────────────────────────────────────────────

def _thumbnail(frame_b64: str) -> np.ndarray | None:
    """Blurred 160x120 grayscale thumbnail for still/changed tests — decoded
    at 1/4 scale straight from the JPEG, so it stays cheap on camera threads."""
    import cv2
    try:
        buf = np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8)
        gray = cv2.imdecode(buf, cv2.IMREAD_REDUCED_GRAYSCALE_4)
    except Exception:
        return None
    if gray is None:
        return None
    return cv2.GaussianBlur(cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA), (5, 5), 0)


def _changed_frac(a: np.ndarray, b: np.ndarray, pixel_delta: int) -> float:
    import cv2
    return float(np.count_nonzero(cv2.absdiff(a, b) > pixel_delta)) / a.size


def clarity(bgr: np.ndarray) -> tuple[float, float]:
    """(mean brightness, sharpness) — sharpness is the Laplacian variance of
    a 640px-wide grayscale copy, so it doesn't depend on camera resolution."""
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] != 640:
        gray = cv2.resize(gray, (640, max(1, round(gray.shape[0] * 640 / gray.shape[1]))),
                          interpolation=cv2.INTER_AREA)
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── the watcher ──────────────────────────────────────────────────────────────

@dataclass
class _CameraState:
    last_sample_ts: float = 0.0
    prev: np.ndarray | None = None
    still_since: float | None = None
    last_read: np.ndarray | None = None  # thumbnail of the last view OCR'd
    busy: bool = False


def _latest_frame(camera: str) -> dict | None:
    from mcp_robot import camera as cam_mod
    cache = {"pi_camera": cam_mod._pi_cache, "simpleipcamera": cam_mod._simpleipcamera_cache}.get(camera)
    return cache.latest() if cache is not None else None


@dataclass
class SceneTextWatcher:
    cameras: tuple[str, ...] = config.SCENE_TEXT_CAMERAS
    registry: SeenTextRegistry = field(default_factory=SeenTextRegistry)
    reader: Callable[[np.ndarray], list[TextLine]] = read_text
    emit: Callable[[dict], None] | None = None  # default: get_signals().add
    context_frame: Callable[[str], dict | None] = _latest_frame
    sample_interval_s: float = config.SCENE_TEXT_SAMPLE_INTERVAL_S
    stable_s: float = config.SCENE_TEXT_STABLE_S
    pixel_delta: int = config.SCENE_TEXT_PIXEL_DELTA
    motion_frac: float = config.SCENE_TEXT_MOTION_FRAC
    change_frac: float = config.SCENE_TEXT_CHANGE_FRAC
    min_brightness: float = config.SCENE_TEXT_MIN_BRIGHTNESS
    max_brightness: float = config.SCENE_TEXT_MAX_BRIGHTNESS
    min_sharpness: float = config.SCENE_TEXT_MIN_SHARPNESS
    image_dir: str = config.SCENE_TEXT_DIR
    threaded: bool = True  # False: read synchronously inside on_frame (tests)

    def __post_init__(self) -> None:
        self._state = {camera: _CameraState() for camera in self.cameras}
        self._queue: "queue.Queue[tuple | None]" = queue.Queue()
        self.reads = 0  # OCR passes run (observability/tests)
        if self.threaded:
            threading.Thread(target=self._work_loop, name="scene-text-ocr", daemon=True).start()

    def on_frame(self, camera: str, frame_b64: str, ts: float) -> None:
        """Called for every frame of every camera; must stay cheap."""
        state = self._state.get(camera)
        # 5% slack: frame timestamps jitter, and one landing a hair short of
        # the interval shouldn't push the sample a whole frame later.
        if state is None or ts - state.last_sample_ts < self.sample_interval_s * 0.95:
            return
        prev_ts, state.last_sample_ts = state.last_sample_ts, ts
        thumb = _thumbnail(frame_b64)
        if thumb is None:
            return
        prev, state.prev = state.prev, thumb
        if prev is None or prev.shape != thumb.shape or _changed_frac(prev, thumb, self.pixel_delta) > self.motion_frac:
            state.still_since = None
            return
        if state.still_since is None:
            state.still_since = prev_ts  # unchanged since the previous sample
        if ts - state.still_since < self.stable_s or state.busy:
            return
        if (state.last_read is not None and state.last_read.shape == thumb.shape
                and _changed_frac(state.last_read, thumb, self.pixel_delta) < self.change_frac):
            return  # this view was already read
        state.busy = True
        if self.threaded:
            self._queue.put((camera, frame_b64, ts, thumb))
        else:
            self._read_view(camera, frame_b64, ts, thumb)

    def _work_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            self._read_view(*item)

    def stop(self) -> None:
        self._queue.put(None)

    def reset(self) -> None:
        """Forget seen words and read views: text in view now counts as new again."""
        self.registry.clear()
        for state in self._state.values():
            state.last_read = None

    def _read_view(self, camera: str, frame_b64: str, ts: float, thumb: np.ndarray) -> None:
        import cv2
        state = self._state[camera]
        try:
            state.last_read = thumb  # whatever comes of it, don't re-read this same view
            bgr = cv2.imdecode(np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return
            brightness, sharpness = clarity(bgr)
            if not (self.min_brightness <= brightness <= self.max_brightness) or sharpness < self.min_sharpness:
                log.debug("scene_text: %s view not clear (brightness=%.0f sharpness=%.1f) — skipped",
                          camera, brightness, sharpness)
                return
            started = time.monotonic()
            lines = self.reader(bgr)
            self.reads += 1
            new = self.registry.novel(lines)
            log.info("scene_text: %s read %d line(s) in %.2fs: %s%s", camera, len(lines),
                     time.monotonic() - started, [l.text for l in lines],
                     f" — NEW: {[l.text for l in new]}" if new else "")
            if not new:
                return
            self.registry.remember(lines)
            signal = self._make_signal(camera, frame_b64, ts, bgr, lines, new)
            (self.emit or get_signals().add)(signal)
        except Exception:
            log.exception("scene_text: reading %s failed", camera)
        finally:
            state.busy = False

    def _make_signal(self, camera, frame_b64, ts, bgr, lines, new) -> dict:
        import cv2
        # Stamped with this host's clock, not the frame's (Pi frame timestamps
        # come from the Pi's own clock).
        now = time.time()
        signal_id = time.strftime("%Y%m%dT%H%M%S", time.localtime(now)) + f"{int(now % 1 * 1000):03d}-{camera}"
        os.makedirs(self.image_dir, exist_ok=True)

        frame_path = os.path.join(self.image_dir, f"{signal_id}.jpg")
        with open(frame_path, "wb") as fh:
            fh.write(base64.b64decode(frame_b64))

        # For reviewing what was read (red = new); send the plain frame to a
        # model, since the boxes cover the text.
        overlay_path = os.path.join(self.image_dir, f"{signal_id}_ocr.jpg")
        overlay = bgr.copy()
        for line in lines:
            cv2.polylines(overlay, [np.array(line.box, dtype=np.int32)], True,
                          (0, 0, 255) if line in new else (0, 200, 0), 2)
        cv2.imwrite(overlay_path, overlay)

        context_path = None
        other = next((c for c in CAMERA_DESCRIPTIONS if c != camera), None)
        frame = self.context_frame(other) if other else None
        if frame and frame.get("frame"):
            context_path = os.path.join(self.image_dir, f"{signal_id}_{other}.jpg")
            with open(context_path, "wb") as fh:
                fh.write(base64.b64decode(frame["frame"]))

        return {
            "from": "neil",
            "type": "visual_signal",
            "id": signal_id,
            "time": time.strftime("%H:%M:%S", time.localtime(now)),
            "camera": camera,
            "texts": [line.text for line in new],  # the lines with a word not seen before
            "all_texts": [line.text for line in lines],
            "frame": frame_path,
            "context_frame": context_path,  # the other camera, same moment
            "ocr_overlay": overlay_path,
            "ts": now,
            "frame_ts": ts,
        }


# ── signals ──────────────────────────────────────────────────────────────────

class SignalBox:
    """Visual signals for the coordinator. Each is appended as one JSON line
    to the outbox — a background `tail -F` of it (Claude Code's Monitor)
    delivers it between tool calls — and stays pending until a tool result
    has carried it (take()). While any is pending the robot is paused: see
    the module docstring."""

    def __init__(self, outbox: str = config.SIGNALS_OUTBOX, recent: int = 20) -> None:
        self.outbox = Path(outbox)
        self._pending: list[dict] = []
        self._recent: deque[dict] = deque(maxlen=recent)
        self._lock = threading.Lock()

    def add(self, signal: dict) -> None:
        with self._lock:
            self._pending.append(signal)
            self._recent.append(signal)
        try:
            self.outbox.parent.mkdir(parents=True, exist_ok=True)
            with open(self.outbox, "a") as fh:  # one write per line, so a tail never sees half of one
                fh.write(json.dumps(signal) + "\n")
        except OSError:
            log.exception("scene_text: could not append to %s", self.outbox)
        log.info("scene_text: visual signal %s from %s: %s", signal["id"], signal["camera"], signal["texts"])

    def pending(self) -> list[dict]:
        with self._lock:
            return list(self._pending)

    def take(self) -> list[dict]:
        """The pending signals, now counted as shown to the coordinator."""
        with self._lock:
            taken, self._pending = self._pending, []
            return taken

    def recent(self) -> list[dict]:
        with self._lock:
            return list(self._recent)


# ── module singletons + hooks used by camera.py / server.py ──────────────────

_singleton_lock = threading.Lock()
_signals: SignalBox | None = None
_watcher: SceneTextWatcher | None = None


def get_signals() -> SignalBox:
    global _signals
    with _singleton_lock:
        if _signals is None:
            _signals = SignalBox()
        return _signals


def get_watcher() -> SceneTextWatcher | None:
    """The live-stream watcher, created on the first frame; None when
    SCENE_TEXT_ENABLED is off."""
    global _watcher
    if not config.SCENE_TEXT_ENABLED:
        return None
    with _singleton_lock:
        if _watcher is None:
            _watcher = SceneTextWatcher()
            log.info("scene_text: watching %s → %s", ", ".join(_watcher.cameras), config.SIGNALS_OUTBOX)
        return _watcher


def on_frame(camera: str, frame_b64: str, ts: float) -> None:
    """Camera-cache hook — never lets a scene-text problem break a stream."""
    try:
        watcher = get_watcher()
        if watcher is not None:
            watcher.on_frame(camera, frame_b64, ts)
    except Exception:
        log.exception("scene_text.on_frame failed")


def should_pause() -> bool:
    """A visual signal is waiting to be shown to the coordinator (and
    pausing is on) — motion must stop, or not start."""
    return config.SCENE_TEXT_PAUSE and bool(get_signals().pending())


_PAUSE_MESSAGE = (
    "PAUSED — the cameras read new text (visual_signals below; look at each `frame`). "
    "Decide whether it changes what the plan should do. If not, re-issue this command to proceed."
)


def paused_result(returns_list: bool):
    """The result of a motor command that didn't run because a visual
    signal was pending — carrying the signals, which counts as showing them."""
    body = {"ok": False, "paused": True, "error": _PAUSE_MESSAGE, "visual_signals": get_signals().take()}
    if returns_list:
        from mcp.types import TextContent
        return [TextContent(type="text", text=json.dumps(body, indent=1))]
    return body


def attach_signals(result):
    """Put visual signals no result has carried yet in front of a tool's
    result — first, so they are read before anything else in it."""
    if not isinstance(result, (dict, list, str)):
        return result  # nowhere to put them — they stay pending for the next result
    signals = get_signals().take()
    if not signals:
        return result
    if isinstance(result, dict):
        return {"visual_signals": signals, **result}
    text = "NEW VISUAL SIGNAL — the cameras read new text; look at each `frame`:\n" + json.dumps(signals, indent=1)
    if isinstance(result, list):
        from mcp.types import TextContent
        return [TextContent(type="text", text=text), *result]
    return f"{text}\n\n{result}"

