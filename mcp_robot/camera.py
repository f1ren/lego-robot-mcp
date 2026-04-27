"""
Camera helpers: capture a still or a short video clip from the RPi OV5647.

All capture happens on the RPi via SSH. Frames are returned as base64 JPEG
strings so they can be embedded directly in MCP ImageContent responses.

When stream_live() is running, capture_still() and capture_clip() read from
the shared frame cache instead of opening a second picamera2 session.
"""
import base64
import time
import threading
from mcp_robot import config, viz
from mcp_robot.rpi_client import get_client


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


_pi_cache = _PiFrameCache()

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


# ── public API ────────────────────────────────────────────────────────────────

def capture_still() -> dict:
    """
    Capture one JPEG frame from the RPi camera.

    If stream_live() is running, returns the latest cached frame to avoid
    opening a second picamera2 session. Falls back to a fresh SSH capture.

    Returns:
        {"frame": "<base64>", "width": int, "height": int, "bytes": int}
    """
    cached = _pi_cache.latest()
    if cached is not None:
        viz.log_still(cached)
        return cached

    script = _CAPTURE_STILL.format(
        w=config.CAMERA_WIDTH,
        h=config.CAMERA_HEIGHT,
        warmup=config.CAMERA_WARMUP,
    )
    result = get_client().run_python(script, timeout=15)
    viz.log_still(result)
    return result


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
    viz.log_clip(result)
    return result


def stream_live(
    fps: float = 5.0,
    on_frame=None,
    stop_event: threading.Event | None = None,
) -> None:
    """
    Stream frames from the RPi camera until stop_event is set.

    Keeps picamera2 open across frames — no per-frame startup cost.
    Uses a separate SSH connection so robot commands still work concurrently.

    Args:
        fps:        Target capture rate (1–10 practical limit over SSH).
        on_frame:   Optional callback(frame_b64: str, timestamp: float).
                    Defaults to viz.log_frame if not provided.
        stop_event: Set this to stop the stream. If None, runs until SSH drops.
    """
    if on_frame is None:
        on_frame = viz.log_frame

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

    # Pre-flight: detect "DroidCam is Busy" before handing the URL to OpenCV,
    # which would just silently fail to open the stream.
    try:
        import urllib.request
        with urllib.request.urlopen(config.DROIDCAM_URL, timeout=3) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                body = resp.read(512).decode(errors="replace")
                if "busy" in body.lower():
                    raise RuntimeError(
                        f"DroidCam is busy (another client is connected). "
                        f"Close the other viewer and retry. URL: {config.DROIDCAM_URL}"
                    )
    except RuntimeError:
        raise
    except Exception:
        pass  # let OpenCV handle connection errors in the normal path

    cap = cv2.VideoCapture(config.DROIDCAM_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open DroidCam stream at {config.DROIDCAM_URL}")
    try:
        while stop_event is None or not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf.tobytes()).decode()
            on_frame(b64, time.time())
    finally:
        cap.release()
