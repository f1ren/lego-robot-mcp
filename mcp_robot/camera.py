"""
Camera helpers: capture a still or a short video clip from the RPi OV5647.

All capture happens on the RPi via SSH. Frames are returned as base64 JPEG
strings so they can be embedded directly in MCP ImageContent responses.

When stream_live() is running, capture_still() and capture_clip() read from
the shared frame cache instead of opening a second picamera2 session.
"""
import base64
import logging
import os
import time
import threading
from mcp_robot import config, viz
from mcp_robot import heading as _heading
from mcp_robot.rpi_client import get_client

log = logging.getLogger(__name__)

_CV2_ROTATE = {
    90: 0,   # cv2.ROTATE_90_CLOCKWISE
    180: 1,  # cv2.ROTATE_180
    270: 2,  # cv2.ROTATE_90_COUNTERCLOCKWISE
}

def _rotate_droidcam(frame):
    """Rotate a BGR frame per DROIDCAM_ROTATION config (0/90/180/270 CW)."""
    code = _CV2_ROTATE.get(config.DROIDCAM_ROTATION)
    if code is None:
        return frame
    import cv2
    return cv2.rotate(frame, code)


def _snapshot_ts_key() -> str:
    """Timestamp key used in snapshot filenames (also shared by raw/annotated pairs)."""
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}"


def _save_snapshot(
    frame_b64: str,
    label: str,
    index: int | None = None,
    ts_key: str | None = None,
) -> str | None:
    """
    Decode frame_b64 and write it to SNAPSHOT_DIR as a JPEG.

    Returns the file path on success, None if saving is disabled or fails.
    label:   "picamera", "clip", "droidcam", "droidcam_raw", ...
    index:   frame index within a clip (None for stills)
    ts_key:  shared timestamp key so a raw/annotated pair of the same frame
             gets matching filenames (differing only by label). Generated
             fresh if not given.
    """
    if not config.SNAPSHOT_DIR:
        return None
    try:
        os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
        if ts_key is None:
            ts_key = _snapshot_ts_key()
        suffix = f"_f{index:02d}" if index is not None else ""
        path = os.path.join(config.SNAPSHOT_DIR, f"{label}_{ts_key}{suffix}.jpg")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(frame_b64))
        log.info("Snapshot saved: %s", path)
        return path
    except Exception as exc:
        log.warning("Failed to save snapshot: %s", exc)
        return None


# ── Pi Camera frame cache ──────────────────────────────────────────────────────

class _PiFrameCache:
    """Thread-safe ring buffer of frames from the live Pi Camera stream."""
    _BUFFER_S = 30  # seconds of history to keep

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf: list[dict] = []

    def put(self, frame_b64: str, ts: float, width: int, height: int) -> None:
        entry = {
            "frame": frame_b64,
            "ts": ts,
            "width": width,
            "height": height,
            "bytes": len(frame_b64) * 3 // 4,  # base64 → byte estimate
        }
        with self._lock:
            self._buf.append(entry)
            cutoff = ts - self._BUFFER_S
            while self._buf and self._buf[0]["ts"] < cutoff:
                self._buf.pop(0)
        from mcp_robot.recorder import get_recorder
        get_recorder().on_frame("pi_camera", frame_b64, ts, cache=self)

    def latest(self) -> dict | None:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def clip(self, duration_s: float, fps: float) -> list[dict] | None:
        """Return a subsampled slice of the buffer, or None if empty."""
        with self._lock:
            if not self._buf:
                return None
            cutoff = time.time() - duration_s
            frames = [f for f in self._buf if f["ts"] >= cutoff] or list(self._buf)
            target_n = max(1, round(duration_s * fps))
            if len(frames) <= target_n:
                return list(frames)
            indices = [round(i * (len(frames) - 1) / (target_n - 1)) for i in range(target_n)]
            return [frames[i] for i in indices]

    def clip_since(self, t_start: float, max_fps: float = 3.0) -> list[dict] | None:
        """Return frames captured since t_start, subsampled to max_fps. None if empty."""
        with self._lock:
            frames = [f for f in self._buf if f["ts"] >= t_start]
            if not frames:
                return None
            duration = frames[-1]["ts"] - frames[0]["ts"]
            target_n = max(2, round(duration * max_fps)) if duration > 0 else len(frames)
            if len(frames) <= target_n:
                return list(frames)
            indices = [round(i * (len(frames) - 1) / (target_n - 1)) for i in range(target_n)]
            return [frames[i] for i in indices]


_pi_cache = _PiFrameCache()


# ── DroidCam frame cache ──────────────────────────────────────────────────────

class _DroidCamFrameCache:
    """Thread-safe ring buffer for DroidCam frames."""
    _BUFFER_S = 30

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf: list[dict] = []

    def put(self, frame_b64: str, ts: float) -> None:
        entry = {
            "frame": frame_b64,
            "ts": ts,
            "bytes": len(frame_b64) * 3 // 4,
        }
        with self._lock:
            self._buf.append(entry)
            cutoff = ts - self._BUFFER_S
            while self._buf and self._buf[0]["ts"] < cutoff:
                self._buf.pop(0)
        from mcp_robot.recorder import get_recorder
        get_recorder().on_frame("droidcam", frame_b64, ts, cache=self)

    def latest(self) -> dict | None:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def clip(self, duration_s: float, fps: float) -> list[dict] | None:
        """Return a subsampled slice of the buffer, or None if empty."""
        with self._lock:
            if not self._buf:
                return None
            cutoff = time.time() - duration_s
            frames = [f for f in self._buf if f["ts"] >= cutoff] or list(self._buf)
            target_n = max(1, round(duration_s * fps))
            if len(frames) <= target_n:
                return list(frames)
            indices = [round(i * (len(frames) - 1) / (target_n - 1)) for i in range(target_n)]
            return [frames[i] for i in indices]

    def clip_since(self, t_start: float, max_fps: float = 3.0) -> list[dict] | None:
        """Return frames captured since t_start, subsampled to max_fps. None if empty."""
        with self._lock:
            frames = [f for f in self._buf if f["ts"] >= t_start]
            if not frames:
                return None
            duration = frames[-1]["ts"] - frames[0]["ts"]
            target_n = max(2, round(duration * max_fps)) if duration > 0 else len(frames)
            if len(frames) <= target_n:
                return list(frames)
            indices = [round(i * (len(frames) - 1) / (target_n - 1)) for i in range(target_n)]
            return [frames[i] for i in indices]


_droidcam_cache = _DroidCamFrameCache()

# ── RPi-side scripts ──────────────────────────────────────────────────────────

_CAPTURE_STILL = """
import json, base64, io, time
from picamera2 import Picamera2
from libcamera import Transform

cam = Picamera2()
cam.configure(cam.create_still_configuration(
    main={{'size': ({w}, {h})}},
    transform=Transform(hflip=True, vflip=True),
))
cam.start()
time.sleep({warmup})
buf = io.BytesIO()
cam.capture_file(buf, format='jpeg')
cam.stop()
cam.close()
print(json.dumps({{
    'frame': base64.b64encode(buf.getvalue()).decode(),
    'width': {w},
    'height': {h},
    'bytes': len(buf.getvalue()),
}}))
"""

_CAPTURE_CLIP = """
import json, base64, io, time
from picamera2 import Picamera2
from PIL import Image
from libcamera import Transform

cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={{'size': ({w}, {h}), 'format': 'RGB888'}},
    transform=Transform(hflip=True, vflip=True),
))
cam.start()
time.sleep({warmup})

frames = []
for _ in range({n_frames}):
    arr = cam.capture_array()
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=82)
    frames.append(base64.b64encode(buf.getvalue()).decode())
    time.sleep({interval})

cam.stop()
cam.close()
print(json.dumps({{
    'frames': frames,
    'count': len(frames),
    'width': {w},
    'height': {h},
}}))
"""


_STREAM_FRAMES = """
import json, base64, io, time
from picamera2 import Picamera2
from PIL import Image
from libcamera import Transform

fps = {fps}
cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={{'size': ({w}, {h}), 'format': 'RGB888'}},
    transform=Transform(hflip=True, vflip=True),
))
cam.start()
time.sleep({warmup})

interval = 1.0 / fps
try:
    while True:
        t0 = time.monotonic()
        arr = cam.capture_array()
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='JPEG', quality=75)
        print(json.dumps({{
            'frame': base64.b64encode(buf.getvalue()).decode(),
            'ts': time.time(),
            'width': {w},
            'height': {h},
        }}), flush=True)
        slack = interval - (time.monotonic() - t0)
        if slack > 0:
            time.sleep(slack)
except (BrokenPipeError, KeyboardInterrupt):
    pass
finally:
    cam.stop()
    cam.close()
"""

# MJPEG HTTP server script — runs on the RPi and serves frames at the sensor's
# native rate (~15-30 fps at 640×480). Read by _stream_live_http via OpenCV.
_PICAMERA_MJPEG_SERVER = """
import io, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
from libcamera import Transform

class _Buf(io.BufferedIOBase):
    def __init__(self):
        self.frame = b''
        self.cond = threading.Condition()
    def write(self, buf):
        with self.cond:
            self.frame = bytes(buf)
            self.cond.notify_all()
        return len(buf)

_buf = _Buf()

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()
        try:
            while True:
                with _buf.cond:
                    _buf.cond.wait()
                    frame = _buf.frame
                hdr = (b'--FRAME\\r\\nContent-Type: image/jpeg\\r\\nContent-Length: '
                       + str(len(frame)).encode() + b'\\r\\n\\r\\n')
                self.wfile.write(hdr + frame + b'\\r\\n')
        except Exception:
            pass
    def log_message(self, *a): pass

cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={{'size': ({w}, {h}), 'format': 'RGB888'}},
    transform=Transform(hflip=True, vflip=True),
))
cam.start_recording(MJPEGEncoder(), FileOutput(_buf))
try:
    HTTPServer(('', {port}), _Handler).serve_forever()
finally:
    cam.stop_recording()
    cam.close()
"""


# ── public API ────────────────────────────────────────────────────────────────

def capture_still() -> dict:
    """
    Capture one JPEG frame from the RPi camera.

    If stream_live() is running, returns the latest cached frame to avoid
    opening a second picamera2 session. Falls back to a fresh SSH capture.

    Returns:
        {"frame": "<base64>", "width": int, "height": int, "bytes": int, "path": str | None}
    """
    cached = _pi_cache.latest()
    if cached is not None:
        path = _save_snapshot(cached["frame"], "picamera")
        viz.log_still(cached)
        return {**cached, "path": path}

    script = _CAPTURE_STILL.format(
        w=config.CAMERA_WIDTH,
        h=config.CAMERA_HEIGHT,
        warmup=config.CAMERA_WARMUP,
    )
    result = get_client().run_python(script, timeout=15)
    path = _save_snapshot(result["frame"], "picamera")
    viz.log_still(result)
    return {**result, "path": path}


def capture_clip(duration_s: float = 2.0, fps: float = 2.0) -> dict:
    """
    Return a short clip as a list of JPEG frames.

    If stream_live() is running, slices the last `duration_s` seconds from
    the frame cache (no extra picamera2 session needed). Falls back to a
    fresh SSH capture otherwise.

    Returns:
        {"frames": ["<base64>", ...], "count": int, "width": int, "height": int}
    """
    clip_frames = _pi_cache.clip(duration_s, fps)
    if clip_frames is not None:
        result = {
            "frames": [f["frame"] for f in clip_frames],
            "count": len(clip_frames),
            "width": clip_frames[0]["width"],
            "height": clip_frames[0]["height"],
        }
        paths = [_save_snapshot(f, "clip", index=i) for i, f in enumerate(result["frames"])]
        result["paths"] = paths
        viz.log_clip(result)
        return result

    n_frames = max(1, round(duration_s * fps))
    interval = 1.0 / fps
    script = _CAPTURE_CLIP.format(
        w=config.CAMERA_WIDTH,
        h=config.CAMERA_HEIGHT,
        warmup=config.CAMERA_WARMUP,
        n_frames=n_frames,
        interval=interval,
    )
    timeout = int(duration_s + 10)
    result = get_client().run_python(script, timeout=timeout)
    paths = [_save_snapshot(f, "clip", index=i) for i, f in enumerate(result["frames"])]
    result["paths"] = paths
    viz.log_clip(result)
    return result


def stream_live(
    fps: float = 15.0,
    on_frame=None,
    stop_event: threading.Event | None = None,
) -> None:
    """
    Stream frames from the RPi camera until stop_event is set.

    Tries the fast path first: starts a picamera2 MJPEG HTTP server on the RPi
    and reads it via OpenCV (no per-frame SSH overhead, ~15-30 fps at 640×480).
    Falls back to SSH JSON-line streaming (~5 fps) if the HTTP server cannot be
    reached.

    Args:
        fps:        Target capture rate hint (used by SSH fallback; HTTP path
                    runs at the camera's native rate).
        on_frame:   Optional callback(frame_b64: str, timestamp: float).
                    Defaults to viz.log_frame if not provided.
        stop_event: Set this to stop the stream.
    """
    if on_frame is None:
        on_frame = viz.log_frame

    try:
        _stream_live_http(on_frame, stop_event)
    except Exception as exc:
        log.warning("Pi Camera HTTP stream failed (%s) — falling back to SSH stream", exc)
        _stream_live_ssh(fps, on_frame, stop_event)


def _stream_live_http(on_frame, stop_event: threading.Event | None) -> None:
    """Fast path: picamera2 MJPEG HTTP server read by OpenCV."""
    import cv2

    url = f"http://{config.RPI_HOST}:{config.PICAMERA_MJPEG_PORT}/"
    server_stop = threading.Event()

    def _run_server() -> None:
        import paramiko as _pm

        script = _PICAMERA_MJPEG_SERVER.format(
            w=config.CAMERA_WIDTH,
            h=config.CAMERA_HEIGHT,
            port=config.PICAMERA_MJPEG_PORT,
        )
        ssh = _pm.SSHClient()
        ssh.set_missing_host_key_policy(_pm.AutoAddPolicy())
        ssh.connect(
            config.RPI_HOST,
            username=config.RPI_USER,
            timeout=config.SSH_TIMEOUT,
            look_for_keys=True,
            allow_agent=True,
        )
        ssh.get_transport().sock.settimeout(None)
        _, _so, _ = ssh.exec_command(
            "fuser -k -TERM /dev/video0 /dev/video1 /dev/media0 2>/dev/null; sleep 0.4"
        )
        _so.channel.recv_exit_status()
        stdin, stdout, stderr = ssh.exec_command("python3 -", timeout=None)
        stdin.write(script.encode())
        stdin.channel.shutdown_write()
        try:
            while not server_stop.is_set():
                if stdout.channel.exit_status_ready():
                    err = stderr.read().decode(errors="replace").strip()
                    log.warning("Pi Camera MJPEG server exited unexpectedly. stderr: %s", err or "(empty)")
                    break
                server_stop.wait(timeout=0.5)
        finally:
            stdin.channel.close()
            ssh.close()

    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    # Poll until the MJPEG server is accepting connections (up to 8 s).
    # SSH + fuser + picamera2 + MJPEGEncoder init can take 3-5 s total.
    time.sleep(1.5)
    cap = None
    deadline = time.time() + 6.5
    while time.time() < deadline:
        c = cv2.VideoCapture(url)
        if c.isOpened():
            cap = c
            break
        c.release()
        time.sleep(0.5)

    if cap is None:
        server_stop.set()
        server_thread.join(timeout=5)
        raise RuntimeError(f"Cannot connect to Pi Camera MJPEG server at {url}")

    log.info("Pi Camera MJPEG stream started at %s", url)
    try:
        while stop_event is None or not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf.tobytes()).decode()
            ts = time.time()
            _pi_cache.put(b64, ts, frame.shape[1], frame.shape[0])
            on_frame(b64, ts)
    finally:
        cap.release()
        server_stop.set()
        server_thread.join(timeout=5)


def _stream_live_ssh(fps: float, on_frame, stop_event: threading.Event | None) -> None:
    """Slow-path fallback: SSH JSON-line stream (~5 fps)."""
    script = _STREAM_FRAMES.format(
        w=config.CAMERA_WIDTH,
        h=config.CAMERA_HEIGHT,
        warmup=config.CAMERA_WARMUP,
        fps=fps,
    )

    def _on_line(data: dict) -> None:
        if "frame" in data:
            _pi_cache.put(
                data["frame"],
                data.get("ts", time.time()),
                data.get("width", config.CAMERA_WIDTH),
                data.get("height", config.CAMERA_HEIGHT),
            )
            on_frame(data["frame"], data.get("ts", 0.0))

    get_client().stream_python(script, _on_line, stop_event)


def _droidcam_failure_reason() -> str:
    """Best-effort diagnosis for why cv2.VideoCapture(config.DROIDCAM_URL) failed.

    Queries mjpeg_bridge's /health endpoint instead of probing
    config.DROIDCAM_URL directly — cap.isOpened()==False alone can't tell
    apart "the bridge process isn't running" from "it's running but the
    phone hasn't opened the capture page yet".
    """
    import json
    import urllib.error
    import urllib.request

    health_url = f"http://127.0.0.1:{config.MJPEG_BRIDGE_HTTP_PORT}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=3) as resp:
            info = json.loads(resp.read())
    except Exception as exc:
        return (f"Cannot reach mjpeg_bridge's /health at {health_url} ({exc}). "
                 f"Is it running? (.venv/bin/python3 -m mcp_robot.mjpeg_bridge)  "
                 f"Raw capture URL: {config.DROIDCAM_URL}")

    if not info.get("phone_connected", False):
        return (f"mjpeg_bridge has no phone connected — open the capture page at "
                 f"https://<this-host>:{config.MJPEG_BRIDGE_WSS_PORT}/ in Safari and tap "
                 f"'Start publishing'. (DROIDCAM_URL={config.DROIDCAM_URL})")
    age = info.get("last_frame_age_s")
    if age is None or age > 5.0:
        return (f"mjpeg_bridge has a connected phone but no recent frames "
                 f"(last_frame_age_s={age}) — publishing may have stalled; check the "
                 f"capture page is still in the foreground. (DROIDCAM_URL={config.DROIDCAM_URL})")
    return (f"mjpeg_bridge reports a connected phone with recent frames "
             f"(last_frame_age_s={age:.1f}), but cv2.VideoCapture still failed to open "
             f"{config.DROIDCAM_URL} — check the mjpeg_bridge process's own logs.")


def stream_droidcam(
    stop_event: threading.Event | None = None,
    on_frame=None,
) -> None:
    """
    Stream frames from DroidCam over HTTP until stop_event is set.

    Reads from config.DROIDCAM_URL using OpenCV (no SSH needed).

    Args:
        stop_event: Set this to stop the stream.
        on_frame:   Optional callback(frame_b64: str, timestamp: float).
                    Defaults to viz.log_droidcam_frame.
    """
    import cv2

    if on_frame is None:
        on_frame = viz.log_droidcam_frame

    cap = cv2.VideoCapture(config.DROIDCAM_URL)
    if not cap.isOpened():
        # Probe the URL to distinguish "busy" from a real connection failure.
        # Done only on failure — probing before VideoCapture opens triggers
        # DroidCam's single-client lockout and breaks the next connect.
        raise RuntimeError(_droidcam_failure_reason())
    try:
        while stop_event is None or not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame = _rotate_droidcam(frame)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf.tobytes()).decode()
            ts = time.time()
            _droidcam_cache.put(b64, ts)
            on_frame(b64, ts)
    finally:
        cap.release()


def capture_droidcam_clip(duration_s: float = 2.0, fps: float = 2.0, annotate: bool = True) -> dict:
    """
    Return a short DroidCam clip as a list of JPEG frames.

    Reads from the cache populated by stream_droidcam(). If no stream is
    running, opens a short-lived cv2.VideoCapture to grab frames directly.

    Args:
        annotate: If True (default), overlay the heading arrow on each frame.
                  Pass False when frames are destined for VLM motion analysis.

    Returns:
        {"frames": ["<base64>", ...], "raw_frames": ["<base64>", ...], "count": int,
         "paths": [...], "raw_paths": [...]}
        raw_frames always contains unannotated frames for VLM use. raw_paths
        mirrors paths but points at the on-disk unannotated copy (same as
        paths when annotate=False, since there's nothing to tell apart then).
    """
    def _maybe_annotate(b64: str) -> str:
        return _heading.annotate_jpeg_b64(b64) if annotate else b64

    def _save_pair(raw_b64: str, display_b64: str, index: int) -> tuple[str | None, str | None]:
        ts_key = _snapshot_ts_key()
        path = _save_snapshot(display_b64, "droidcam_clip", index=index, ts_key=ts_key)
        if not annotate:
            return path, path
        raw_path = _save_snapshot(raw_b64, "droidcam_clip_raw", index=index, ts_key=ts_key)
        return path, raw_path

    clip_frames = _droidcam_cache.clip(duration_s, fps)
    if clip_frames is not None:
        raw = [f["frame"] for f in clip_frames]
        display = [_maybe_annotate(f) for f in raw]
        paths: list[str | None] = []
        raw_paths: list[str | None] = []
        for i, (r, d) in enumerate(zip(raw, display)):
            path, raw_path = _save_pair(r, d, i)
            paths.append(path)
            raw_paths.append(raw_path)
        return {"frames": display, "raw_frames": raw, "count": len(display),
                "paths": paths, "raw_paths": raw_paths}

    import cv2

    cap = cv2.VideoCapture(config.DROIDCAM_URL)
    if not cap.isOpened():
        raise RuntimeError(_droidcam_failure_reason())
    try:
        n_frames = max(1, round(duration_s * fps))
        interval = 1.0 / fps
        raw_frames: list[str] = []
        display_frames: list[str] = []
        paths = []
        raw_paths = []
        for i in range(n_frames):
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            frame = _rotate_droidcam(frame)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            raw = base64.b64encode(buf.tobytes()).decode()
            display = _maybe_annotate(raw)
            raw_frames.append(raw)
            display_frames.append(display)
            path, raw_path = _save_pair(raw, display, i)
            paths.append(path)
            raw_paths.append(raw_path)
            slack = interval - (time.time() - t0)
            if slack > 0 and i < n_frames - 1:
                time.sleep(slack)
        return {"frames": display_frames, "raw_frames": raw_frames, "count": len(display_frames),
                "paths": paths, "raw_paths": raw_paths}
    finally:
        cap.release()


def stream_droidcam_bgr(
    on_frame,
    stop_event: threading.Event,
) -> None:
    """Deliver a continuous stream of BGR frames from DroidCam.

    If _droidcam_cache already holds frames newer than 2 s (meaning
    stream_droidcam() is running), polls the cache so we don't open a
    competing VideoCapture connection (DroidCam only allows one client).
    Otherwise opens a short-lived VideoCapture for the duration of the call.

    on_frame(bgr: np.ndarray, ts: float) is called for each new frame.
    Returns when stop_event is set or the stream ends.
    """
    import cv2 as _cv2
    import numpy as _np

    entry = _droidcam_cache.latest()
    use_cache = entry is not None and (time.time() - entry.get("ts", 0.0)) < 2.0

    if use_cache:
        last_ts = 0.0
        while not stop_event.is_set():
            e = _droidcam_cache.latest()
            if e is None or e.get("ts", 0.0) <= last_ts:
                time.sleep(0.03)
                continue
            last_ts = e["ts"]
            raw = base64.b64decode(e["frame"])
            arr = _np.frombuffer(raw, dtype=_np.uint8)
            bgr = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
            if bgr is not None:
                on_frame(bgr, last_ts)
        return

    cap = _cv2.VideoCapture(config.DROIDCAM_URL)
    if not cap.isOpened():
        log.warning("stream_droidcam_bgr: cannot open DroidCam at %s", config.DROIDCAM_URL)
        return
    try:
        while not stop_event.is_set():
            ok, bgr = cap.read()
            if not ok:
                break
            on_frame(_rotate_droidcam(bgr), time.time())
    finally:
        cap.release()


def capture_droidcam_still(
    target_class_yolo: str,
    annotate: bool = True,
    target_class_free_text: str = "",
) -> dict:
    """
    Return the most recent DroidCam frame.

    DroidCam allows only one client at a time, so we read from the cache
    populated by stream_droidcam(). If no stream is running, opens a
    short-lived cv2.VideoCapture to grab a single frame. The returned frame
    is overlaid with a green forward-arrow when the robot's heading can be
    detected (see mcp_robot.heading).

    Args:
        target_class_yolo:      YOLO class key to detect (e.g. "cup", "ball", "any").
                                The detected object's angle relative to the robot's
                                heading is included in the returned dict. Pass ""
                                to skip object detection (heading arrow only, or
                                no annotation at all if annotate=False). Every
                                caller must decide explicitly — there is no
                                sensible universal default.
        annotate:               If True (default), overlay the heading arrow. Pass
                                False when the frame is destined for VLM change
                                analysis to avoid the arrow causing spurious motion
                                detections.
        target_class_free_text: Free-text description for Gemini Flash fallback when
                                YOLO finds nothing (e.g. "light switch").

    Returns:
        {"frame": "<base64>", "ts": float, "bytes": int, "path": str | None,
         "raw_path": str | None,
         "object_angle_deg": float | None, "vlm_note": str,
         "object_distance_px": float | None, "robot_radius_px": float | None,
         "robot_body_area_px": int | None}
        raw_path points at the on-disk unannotated copy (same as path when
        annotate=False). Kept alongside the annotated copy so a bad heading
        arrow or object angle can be diagnosed against the clean frame.
    """
    def _maybe_annotate(b64: str) -> tuple[str, float | None, str, float | None, float | None, int | None]:
        if not annotate:
            return b64, None, "", None, None, None
        from mcp_robot import grasp_readiness as _gr
        annotated_b64, angle_deg, note, dist_px, robot_radius_px, body_area_px = _gr.annotate_frame_with_object_b64(
            b64,
            target_class_yolo=target_class_yolo,
            target_class_free_text=target_class_free_text,
        )
        if angle_deg is not None:
            rot_dir = "CW" if angle_deg > 0 else "CCW"
            log.info(
                "Heading analysis: yolo=%r free_text=%r — object at %.1f° %s from forward, dist=%.1fpx robot_r=%.1fpx",
                target_class_yolo, target_class_free_text, abs(angle_deg), rot_dir,
                dist_px or 0, robot_radius_px or 0,
            )
        elif robot_radius_px is None:
            # annotate_frame_with_object bails out before YOLO/VLM run at all
            # when detect_heading() itself fails — see mcp_robot.heading logs
            # for which stage (body/axis/gripper) failed.
            log.info(
                "Heading analysis: yolo=%r free_text=%r — robot heading not detected; "
                "object search was not attempted (see mcp_robot.heading log above)",
                target_class_yolo, target_class_free_text,
            )
        else:
            log.info(
                "Heading analysis: yolo=%r free_text=%r — heading OK, but no matching object found "
                "(YOLO + VLM fallback both missed); heading arrow only",
                target_class_yolo, target_class_free_text,
            )
        return annotated_b64, angle_deg, note, dist_px, robot_radius_px, body_area_px

    def _build_result(base: dict, b64: str) -> dict:
        frame, angle_deg, note, dist_px, robot_radius_px, body_area_px = _maybe_annotate(b64)
        ts_key = _snapshot_ts_key()
        path = _save_snapshot(frame, "droidcam", ts_key=ts_key)
        raw_path = _save_snapshot(b64, "droidcam_raw", ts_key=ts_key) if annotate else path
        return {**base, "frame": frame, "path": path, "raw_path": raw_path,
                "object_angle_deg": angle_deg, "vlm_note": note,
                "object_distance_px": dist_px, "robot_radius_px": robot_radius_px,
                "robot_body_area_px": body_area_px}

    cached = _droidcam_cache.latest()
    if cached is not None:
        return _build_result(cached, cached["frame"])

    import cv2

    cap = cv2.VideoCapture(config.DROIDCAM_URL)
    if not cap.isOpened():
        raise RuntimeError(_droidcam_failure_reason())
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"DroidCam read failed at {config.DROIDCAM_URL}")
        frame = _rotate_droidcam(frame)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf.tobytes()).decode()
        return _build_result({"ts": time.time(), "bytes": len(buf)}, b64)
    finally:
        cap.release()
