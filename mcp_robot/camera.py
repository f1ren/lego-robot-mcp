"""
Camera helpers: capture a still or a short video clip from the RPi OV5647.

All capture happens on the RPi via SSH. Frames are returned as base64 JPEG
strings so they can be embedded directly in MCP ImageContent responses.
"""
from mcp_robot import config
from mcp_robot.rpi_client import get_client

# ── RPi-side scripts ──────────────────────────────────────────────────────────

_CAPTURE_STILL = """
import json, base64, io, time
from picamera2 import Picamera2

cam = Picamera2()
cam.configure(cam.create_still_configuration(
    main={{'size': ({w}, {h})}}
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

cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={{'size': ({w}, {h}), 'format': 'RGB888'}}
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


# ── public API ────────────────────────────────────────────────────────────────

def capture_still() -> dict:
    """
    Capture one JPEG frame from the RPi camera.

    Returns:
        {"frame": "<base64>", "width": int, "height": int, "bytes": int}
    """
    script = _CAPTURE_STILL.format(
        w=config.CAMERA_WIDTH,
        h=config.CAMERA_HEIGHT,
        warmup=config.CAMERA_WARMUP,
    )
    return get_client().run_python(script, timeout=15)


def capture_clip(duration_s: float = 2.0, fps: float = 2.0) -> dict:
    """
    Capture a short video clip as a list of JPEG frames.

    Args:
        duration_s: Total clip length in seconds.
        fps:        Frames per second to capture (max ~5 with picamera2 stills).

    Returns:
        {"frames": ["<base64>", ...], "count": int, "width": int, "height": int}
    """
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
    return get_client().run_python(script, timeout=timeout)
