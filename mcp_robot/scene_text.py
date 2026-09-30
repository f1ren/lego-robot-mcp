"""
Scene text: notice new text (signs, labels, warnings) in the live camera
streams, hand it to NAPC — the separate planning MCP server — and surface
NAPC's advisories (halt / new plan) to the coordinator (the MCP caller).

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
                     finding: frames → SCENE_TEXT_DIR,
                              JSON   → SCENE_EVENTS_DIR/findings/<id>.json
                       ▼
                     NAPC judges it against the active plan (triage → halt?
                     → adapt domain/problem → replan) and writes
                     SCENE_EVENTS_DIR/advisories/<id>.json
                       ▼
                     AdvisoryBoard.collect() → "napc_advisory" in every tool
                     result; navigate_to/scans stop early while a halt stands

The two servers share nothing but that directory (see NAPC's
napc/scene_hints.py FILE PROTOCOL): neither is an MCP client of the other,
and the coordinator stays the one that decides and acts.

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

# NAPC advisory statuses (napc/scene_hints.py) the coordinator must act on.
_ACTIONABLE = {"replanning", "new_plan", "unchanged", "failed"}
_ACTIVE = {"queued", "triaging", "replanning"}


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
    emit: Callable[[dict], None] | None = None  # default: post_finding
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
            finding = self._make_finding(camera, frame_b64, ts, bgr, lines, new)
            (self.emit or post_finding)(finding)
        except Exception:
            log.exception("scene_text: reading %s failed", camera)
        finally:
            state.busy = False

    def _make_finding(self, camera, frame_b64, ts, bgr, lines, new) -> dict:
        import cv2
        # Stamped with this host's clock, not the frame's: Pi frame timestamps
        # come from the Pi's own clock, and NAPC compares a finding's "ts"
        # against its own start time to skip leftovers from earlier sessions.
        now = time.time()
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(now)) + f"{int(now % 1 * 1000):03d}"
        finding_id = f"{stamp}-{camera}"
        os.makedirs(self.image_dir, exist_ok=True)
        images = []

        path = os.path.join(self.image_dir, f"{finding_id}.jpg")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(frame_b64))
        images.append({"label": camera, "path": path,
                       "description": f"{CAMERA_DESCRIPTIONS.get(camera, camera)} — the text was read in this view"})

        # For people reviewing findings, not for NAPC: the boxes would cover the text.
        overlay = bgr.copy()
        for line in lines:
            pts = np.array(line.box, dtype=np.int32)
            cv2.polylines(overlay, [pts], True, (0, 0, 255) if line in new else (0, 200, 0), 2)
        cv2.imwrite(os.path.join(self.image_dir, f"{finding_id}_ocr.jpg"), overlay)

        for other in CAMERA_DESCRIPTIONS:
            if other == camera:
                continue
            frame = self.context_frame(other)
            if frame and frame.get("frame"):
                other_path = os.path.join(self.image_dir, f"{finding_id}_{other}.jpg")
                with open(other_path, "wb") as fh:
                    fh.write(base64.b64decode(frame["frame"]))
                images.append({"label": other, "path": other_path,
                               "description": f"{CAMERA_DESCRIPTIONS[other]} — context, same moment"})

        return {
            "schema": 1,
            "finding_id": finding_id,
            "ts": now,
            "frame_ts": ts,
            "source": "lego-robot",
            "camera": camera,
            "texts": [line.text for line in new],
            "all_texts": [line.to_dict() for line in lines],
            "images": images,
        }


# ── NAPC mailbox ─────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


class AdvisoryBoard:
    """Reads NAPC's advisories back from SCENE_EVENTS_DIR/advisories and
    decides which to put in front of the coordinator, on which tool result.

    Each advisory update (its "seq") is surfaced once, except a standing
    halt, which rides on every result until NAPC moves past it — a halt
    matters until it is lifted, not only when first announced."""

    def __init__(self, events_dir: str = config.SCENE_EVENTS_DIR,
                 napc_timeout_s: float = config.SCENE_TEXT_NAPC_TIMEOUT_S,
                 halt_max_age_s: float = config.SCENE_TEXT_HALT_MAX_AGE_S,
                 clock: Callable[[], float] = time.time) -> None:
        self.events_dir = Path(events_dir)
        self.napc_timeout_s = napc_timeout_s
        self.halt_max_age_s = halt_max_age_s
        self._clock = clock
        self._started_at = clock()
        self._posted: dict[str, dict] = {}  # findings this process sent
        self._delivered: dict[str, int] = {}  # finding_id -> seq surfaced
        self._dismissed: dict[str, int] = {}  # finding_id -> seq the coordinator dismissed
        self._unanswered_reported: set[str] = set()
        self._cache: dict[str, tuple[int, dict]] = {}
        self._lock = threading.Lock()

    def post(self, finding: dict) -> None:
        _atomic_write_json(self.events_dir / "findings" / f"{finding['finding_id']}.json", finding)
        with self._lock:
            self._posted[finding["finding_id"]] = finding
        log.info("scene_text: finding %s → NAPC: %s", finding["finding_id"], finding["texts"])

    def _advisories(self) -> dict[str, dict]:
        """Advisories for findings this process sent, or updated since it started."""
        found = {}
        try:
            paths = list((self.events_dir / "advisories").glob("*.json"))
        except OSError:
            return found
        for path in paths:
            try:
                mtime = path.stat().st_mtime_ns
                cached = self._cache.get(path.name)
                if cached is None or cached[0] != mtime:
                    cached = (mtime, json.loads(path.read_text()))
                    self._cache[path.name] = cached
            except (OSError, ValueError):
                continue
            advisory = cached[1]
            fid = advisory.get("finding_id", path.stem)
            if self._dismissed.get(fid) == advisory.get("seq"):
                continue
            if fid in self._posted or advisory.get("updated_at", 0) >= self._started_at:
                found[fid] = advisory
        return found

    def dismiss(self, finding_ids: list[str]) -> list[str]:
        """Stop surfacing (and halting on) these advisories as they stand now —
        the escape hatch for a halt NAPC will never lift (e.g. it stopped
        mid-replan). A later update from NAPC brings an advisory back."""
        with self._lock:
            current = self._advisories()
            for fid in finding_ids:
                if fid in current:
                    self._dismissed[fid] = current[fid].get("seq")
            return [fid for fid in finding_ids if fid in self._dismissed]

    def halt_advisory(self) -> dict | None:
        """The standing halt, if NAPC is replanning and recommends halting."""
        now = self._clock()
        with self._lock:
            for advisory in self._advisories().values():
                if (advisory.get("status") in _ACTIVE and advisory.get("halt")
                        and not advisory.get("acknowledged")
                        and now - advisory.get("updated_at", now) <= self.halt_max_age_s):
                    return advisory
        return None

    def collect(self) -> list[dict]:
        """Advisories to show the coordinator on this tool result."""
        now = self._clock()
        entries = []
        with self._lock:
            advisories = self._advisories()
            for fid, advisory in sorted(advisories.items(), key=lambda kv: kv[1].get("updated_at", 0)):
                status, seq = advisory.get("status"), advisory.get("seq")
                if advisory.get("acknowledged") or status not in _ACTIONABLE:
                    continue
                standing_halt = status == "replanning" and advisory.get("halt")
                if standing_halt or self._delivered.get(fid) != seq:
                    self._delivered[fid] = seq
                    entries.append(self._format(advisory, stale=now - advisory.get("updated_at", now) > self.halt_max_age_s))
            for fid, finding in self._posted.items():
                if fid in advisories or fid in self._unanswered_reported or now - finding["ts"] < self.napc_timeout_s:
                    continue
                self._unanswered_reported.add(fid)
                entries.append({
                    "finding_id": fid, "status": "napc_silent", "texts": finding["texts"],
                    "camera": finding["camera"],
                    "images": [i["path"] for i in finding["images"]],
                    "message": f"The {finding['camera']} camera read new text {finding['texts']}, but NAPC has not "
                               f"judged it after {self.napc_timeout_s:.0f}s (is the napc server running with "
                               "NAPC_EVENTS_DIR set?). Judge yourself whether it matters for the current plan; "
                               "if so, call napc report_scene_text or consult_vqa_for_pddl_domain with the images.",
                })
        return entries

    @staticmethod
    def _format(advisory: dict, stale: bool = False) -> dict:
        status = advisory.get("status")
        fid = advisory.get("finding_id")
        entry = {
            "finding_id": fid,
            "status": status,
            "halt": bool(advisory.get("halt")) and status == "replanning",
            "texts": advisory.get("texts"),
            "camera": advisory.get("camera"),
            "summary": advisory.get("summary"),
        }
        if status == "replanning" and advisory.get("halt"):
            entry["action"] = (f"HALT: stop executing the current plan now — NAPC is replanning because of this "
                               f"text. Then call napc await_advisory(finding_id={fid!r}, completed_steps=<steps of "
                               "your current plan fully completed>) and continue with the plan it returns.")
        elif status == "replanning":
            entry["action"] = (f"Finish the current step, then call napc await_advisory(finding_id={fid!r}, "
                               "completed_steps=<steps of your current plan fully completed>) — NAPC is replanning "
                               "because of this text.")
        elif status == "new_plan":
            entry["action"] = (f"NAPC replanned because of this text. Call napc await_advisory(finding_id={fid!r}, "
                               "completed_steps=<steps of your current plan fully completed>) to get the new plan "
                               "resumed from where you are, and continue with it.")
        elif status == "unchanged":
            entry["action"] = "NAPC found nothing to change for this text — continue your current plan."
        else:  # failed
            entry["action"] = advisory.get("message") or "NAPC could not evaluate this text."
            entry["error"] = advisory.get("error")
        if stale:
            entry["note"] = ("NAPC has not updated this advisory for a long time — it may have stopped. Check napc "
                             f"get_advisories; if it is dead, scene_text_status(dismiss=[{fid!r}]) drops it.")
        return entry

    def status(self) -> dict:
        with self._lock:
            return {
                "events_dir": str(self.events_dir),
                "findings_sent": [{"finding_id": f, "texts": d["texts"], "camera": d["camera"]}
                                  for f, d in self._posted.items()],
                "advisories": [{k: a.get(k) for k in ("finding_id", "status", "halt", "texts", "summary")}
                               for a in self._advisories().values()],
            }


# ── module singletons + hooks used by camera.py / server.py ──────────────────

_singleton_lock = threading.Lock()
_board: AdvisoryBoard | None = None
_watcher: SceneTextWatcher | None = None


def get_board() -> AdvisoryBoard:
    global _board
    with _singleton_lock:
        if _board is None:
            _board = AdvisoryBoard()
        return _board


def post_finding(finding: dict) -> None:
    get_board().post(finding)


def get_watcher() -> SceneTextWatcher | None:
    """The live-stream watcher, created on the first frame; None when
    SCENE_TEXT_ENABLED is off."""
    global _watcher
    if not config.SCENE_TEXT_ENABLED:
        return None
    with _singleton_lock:
        if _watcher is None:
            _watcher = SceneTextWatcher()
            log.info("scene_text: watching %s → %s", ", ".join(_watcher.cameras), config.SCENE_EVENTS_DIR)
        return _watcher


def on_frame(camera: str, frame_b64: str, ts: float) -> None:
    """Camera-cache hook — never lets a scene-text problem break a stream."""
    try:
        watcher = get_watcher()
        if watcher is not None:
            watcher.on_frame(camera, frame_b64, ts)
    except Exception:
        log.exception("scene_text.on_frame failed")


def halt_advisory() -> dict | None:
    try:
        return get_board().halt_advisory()
    except Exception:
        log.exception("scene_text: reading advisories failed")
        return None


def attach_advisories(result):
    """Put pending NAPC advisories in front of a tool's result — first, so
    they are read before anything else in it."""
    try:
        entries = get_board().collect()
    except Exception:
        log.exception("scene_text: collecting advisories failed")
        return result
    if not entries:
        return result
    log.info("scene_text: surfacing %s", [(e["finding_id"], e["status"]) for e in entries])
    if isinstance(result, dict):
        return {"napc_advisory": entries, **result}
    text = "NAPC ADVISORY — act on this before anything else below:\n" + json.dumps(entries, indent=1)
    if isinstance(result, list):
        from mcp.types import TextContent
        return [TextContent(type="text", text=text), *result]
    if isinstance(result, str):
        return f"{text}\n\n{result}"
    return result
