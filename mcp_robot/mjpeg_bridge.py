"""
Self-hosted MJPEG-over-HTTP bridge for the iPhone's camera — replaces both
DroidCam and the mediamtx/WHIP bridge that replaced it. See config.py's
"mjpeg_bridge" section for why: mediamtx's WebRTC ingest inherited ordinary
WiFi packet loss as corrupted H264 frames, because a lost RTP packet
corrupts a whole NAL via FU-A fragmentation. Safari has no API to make a web
page act as an HTTP server, so this restores DroidCam's actual wire
behavior (independent frames delivered over TCP, so a lost packet is
retransmitted transparently instead of corrupting a frame) via a different
route: a capture page grabs getUserMedia frames and feeds them to a
WebCodecs VideoEncoder (hardware-accelerated H264, Annex-B byte stream),
pushing each encoded chunk over a WebSocket to this process. An
ffmpeg subprocess (one per connection) decodes that H264 stream and
re-encodes each frame as JPEG, and this process re-serves the latest JPEG
as a standard multipart/x-mixed-replace MJPEG stream — exactly what
cv2.VideoCapture already expects from config.DROIDCAM_URL.

An earlier version of the capture page JPEG-encoded frames directly via
canvas.toBlob(), client-side, in software. That capped out around 17-18fps
on the target iPhone regardless of scheduling — canvas readback/encode is
deliberately kept off the GPU by browsers — because the real cost is the
per-frame software JPEG encode itself, not JS overhead. Hardware H264
encoding via WebCodecs (Safari 26+) sidesteps that ceiling entirely; the
JPEG step still happens, just moved server-side (ffmpeg, one decode+encode
pass, off the phone's CPU).

Requires the `ffmpeg` binary on PATH (used as a subprocess to decode the
incoming H264 and re-encode to JPEG — see the "H264 decode pipeline"
section below).

Run standalone for validation:
    .venv/bin/python3 -m mcp_robot.mjpeg_bridge

Two listeners:
  - HTTPS+WSS on MJPEG_BRIDGE_WSS_PORT: serves the capture page at "/" and
    accepts the WebSocket frame feed at "/ingest". Needs TLS because
    getUserMedia requires a secure context off localhost — reuses the same
    cert already trusted on the phone for mediamtx's old WHIP publish.
  - Plain HTTP on MJPEG_BRIDGE_HTTP_PORT: serves "/video" (the MJPEG stream,
    read by cv2.VideoCapture) and "/health" (JSON status, read by
    camera._droidcam_failure_reason()). No TLS needed — only local
    processes read this.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mcp_robot import config

log = logging.getLogger(__name__)

_BOUNDARY = "frame"
_STALL_TIMEOUT_S = 5.0
_MAX_CONSECUTIVE_STALLS = 6  # ~30s of silence before the /video connection gives up


class _FrameStore:
    """Thread-safe holder for the latest JPEG frame.

    Bridges the asyncio WebSocket-ingest thread and the threading-based
    /video and /health HTTP server — plain threading.Condition works from
    either side since the asyncio handler only ever acquires/notifies
    (never blocks waiting), and the HTTP server's per-connection threads do
    the actual waiting.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._jpeg: "bytes | None" = None
        self._ts: float = 0.0
        self._frames_received = 0
        self._client_connected = False

    def put(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._ts = time.time()
            self._frames_received += 1
            self._cond.notify_all()

    def set_connected(self, connected: bool) -> None:
        with self._lock:
            self._client_connected = connected

    def clear(self) -> None:
        """Drop the cached frame so a vanished publisher reads as "no data"
        immediately, instead of a new /video connection instantly serving a
        stale frame as if it were live (see mjpeg_bridge module docstring's
        history — a lost WiFi/backgrounded-Safari disconnect left cv2 seeing
        one fake-fresh read of minutes-old data before the real stall showed
        up 30s later)."""
        with self._cond:
            self._jpeg = None
            self._ts = 0.0
            self._cond.notify_all()

    def wait_for_frame(self, after_ts: float, timeout: float) -> "tuple[bytes, float] | None":
        """Block until a frame newer than after_ts arrives, or timeout."""
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._jpeg is not None and self._ts > after_ts, timeout=timeout
            ):
                return None
            return self._jpeg, self._ts

    def status(self) -> dict:
        with self._lock:
            age = (time.time() - self._ts) if self._ts else None
            return {
                "phone_connected": self._client_connected,
                "frames_received": self._frames_received,
                "last_frame_age_s": age,
            }


_store = _FrameStore()


# ── capture page (served over HTTPS at "/") ─────────────────────────────────

_CAPTURE_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>lego-robot camera bridge</title>
<style>
  html, body { margin: 0; height: 100%; background: #1e1e1e; color: #fff; font-family: sans-serif; }
  video { width: 100%; height: 65vh; object-fit: contain; background: #000; }
  #controls { padding: 10px; display: flex; flex-direction: column; gap: 8px; }
  label { display: flex; justify-content: space-between; gap: 10px; }
  input, select { width: 140px; }
  #status { padding: 0 10px 10px; font-size: 14px; }
  button { height: 44px; font-size: 16px; }
</style>
</head>
<body>
<video id="video" muted autoplay playsinline></video>
<div id="controls">
  <label>facing <select id="facing"><option value="environment" selected>back</option><option value="user">front</option></select></label>
  <label>width (ideal) <input id="width" value="1280"></label>
  <label>height (ideal) <input id="height" value="720"></label>
  <label>framerate (ideal) <input id="fps" value="30"></label>
  <label>bitrate (kbps) <input id="bitrate" value="4000"></label>
  <button id="start">Start publishing</button>
</div>
<div id="status">idle</div>
<script>
const statusEl = document.getElementById('status');
document.getElementById('start').onclick = async () => {
  const facing = document.getElementById('facing').value;
  const width = parseInt(document.getElementById('width').value, 10);
  const height = parseInt(document.getElementById('height').value, 10);
  const fps = parseInt(document.getElementById('fps').value, 10);
  const bitrate = parseInt(document.getElementById('bitrate').value, 10) * 1000;

  if (typeof VideoEncoder === 'undefined') {
    statusEl.textContent = 'error: WebCodecs (VideoEncoder) not available in this browser';
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: facing },
      width: { ideal: width },
      height: { ideal: height },
      frameRate: { ideal: fps },
    },
    audio: false,
  });
  const video = document.getElementById('video');
  video.srcObject = stream;
  await video.play();

  // The phone's camera delivers portrait frames (video.videoWidth <
  // video.videoHeight) and the local preview renders them correctly — but
  // encoding portrait dimensions directly came out stretched even with a
  // verified-matching encoder config, traced all the way to the raw H264
  // bytes the phone transmits (bypassing this page's own backend entirely
  // still showed it). That points at the hardware encoder itself handling
  // direct portrait encoding badly — probably landscape-native under the
  // hood, like most mobile video hardware. So: rotate into a landscape
  // canvas here and encode *that*; the server rotates back after decode
  // (see mjpeg_bridge.py's _FFMPEG_ARGS "-vf transpose=2").
  const encodeWidth = video.videoHeight;
  const encodeHeight = video.videoWidth;
  const rotateCanvas = document.createElement('canvas');
  rotateCanvas.width = encodeWidth;
  rotateCanvas.height = encodeHeight;
  const rotateCtx = rotateCanvas.getContext('2d');

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ingest`);
  ws.binaryType = 'arraybuffer';
  let sent = 0;
  ws.onclose = () => { statusEl.textContent = 'disconnected'; };
  ws.onerror = () => { statusEl.textContent = 'error - see browser console'; };

  // Baseline profile, level 4.0 ("avc1.420028") comfortably covers
  // 1280x720@30 — level 3.1 sits exactly at its ceiling there (30fps at
  // 720p is the max MaxMBPS/MaxFS allow) — and baseline is the most
  // broadly hardware-supported H264 profile. Annex-B format embeds SPS/PPS
  // inline before each keyframe, so the server's `ffmpeg -f h264` demuxer
  // can decode this as a bare stream with no container and no separate
  // parameter-set channel.
  const encoderConfig = {
    codec: 'avc1.420028',
    width: encodeWidth,
    height: encodeHeight,
    bitrate,
    framerate: fps,
    hardwareAcceleration: 'prefer-hardware',
    avc: { format: 'annexb' },
    latencyMode: 'realtime',
  };
  let support = await VideoEncoder.isConfigSupported(encoderConfig);
  if (!support.supported) {
    encoderConfig.hardwareAcceleration = 'no-preference';
    support = await VideoEncoder.isConfigSupported(encoderConfig);
    if (!support.supported) {
      statusEl.textContent = 'error: no supported VideoEncoder config for avc1.420028';
      return;
    }
  }

  let encoderErrored = false;
  const encoder = new VideoEncoder({
    output: (chunk) => {
      const buf = new ArrayBuffer(chunk.byteLength);
      chunk.copyTo(buf);
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(buf);
        sent++;
        statusEl.textContent = `publishing... (${sent} chunks sent, ${encodeWidth}x${encodeHeight}, ${chunk.type})`;
      }
    },
    error: (e) => {
      encoderErrored = true;
      statusEl.textContent = `encoder error: ${e.message}`;
    },
  });
  encoder.configure(encoderConfig);

  // requestVideoFrameCallback + drawing into the rotated canvas, then
  // constructing the VideoFrame from that canvas instead of straight from
  // <video>. This drawImage is a plain GPU blit+rotate, not a JPEG encode —
  // nothing like the old canvas.toBlob() software-encode bottleneck — so it
  // shouldn't reintroduce the fps ceiling that motivated dropping canvas
  // the first time.
  const onFrame = (_now, metadata) => {
    if (encoderErrored || ws.readyState === WebSocket.CLOSED) return;
    if (encoder.encodeQueueSize > 2) {
      // Hardware encoder is behind — drop this frame rather than let
      // latency grow unboundedly; the robot feed must stay live, not just
      // eventually complete.
      video.requestVideoFrameCallback(onFrame);
      return;
    }
    rotateCtx.save();
    rotateCtx.translate(rotateCanvas.width, 0);
    rotateCtx.rotate(Math.PI / 2);
    rotateCtx.drawImage(video, 0, 0);
    rotateCtx.restore();
    const frame = new VideoFrame(rotateCanvas, { timestamp: Math.round(metadata.mediaTime * 1e6) });
    encoder.encode(frame);
    frame.close();
    video.requestVideoFrameCallback(onFrame);
  };
  // Don't start producing frames until the socket is actually open. The
  // encoder's very first output chunk is the only one carrying inline
  // SPS/PPS (Annex-B puts parameter sets once, before the first keyframe,
  // not repeated per frame) — if that chunk were produced while the socket
  // was still CONNECTING, the send-only-if-OPEN check above would silently
  // drop it with no retry, leaving every later delta frame referencing a
  // PPS the server never received. Then decode never recovers for the rest
  // of the connection (confirmed directly: captured raw bytes from a
  // session that hit this showed zero SPS/PPS/IDR NALs anywhere in it,
  // only delta slices, matching ffmpeg's "non-existing PPS 0" on every
  // frame). Waiting here removes the race instead of working around it.
  await new Promise((resolve) => {
    ws.onopen = () => { statusEl.textContent = 'connected, publishing...'; resolve(); };
  });
  video.requestVideoFrameCallback(onFrame);
};
</script>
</body>
</html>
"""


# ── H264 decode pipeline: ffmpeg subprocess, one per connection ─────────────

_FFMPEG_BIN = shutil.which("ffmpeg")
_FFMPEG_ARGS = [
    _FFMPEG_BIN,
    "-loglevel", "error",
    "-flags", "low_delay",
    "-f", "h264",
    "-i", "pipe:0",
    # The capture page pre-rotates portrait frames 90deg clockwise into a
    # landscape canvas before encoding (see _CAPTURE_PAGE) — encoding
    # portrait dimensions directly came out stretched, traced to the
    # phone's hardware encoder itself. transpose=2 (90deg counterclockwise)
    # undoes that rotation to restore the original portrait orientation.
    "-vf", "transpose=2",
    "-f", "mjpeg",
    "-q:v", "3",
    "pipe:1",
]
# "-fflags nobuffer" here made the h264 decoder emit "Missing reference
# picture" and produce zero output frames on a real Annex-B baseline stream
# (verified directly: piping the same input through ffmpeg by hand, adding
# just -fflags nobuffer took frame count 150 -> 0). -flags low_delay alone
# is fine. Root cause not fully understood (likely starves the demuxer of
# enough buffered bytes to establish reference pictures from a raw
# elementary stream over a pipe) — dropped rather than chased further.


class _MjpegSplitter:
    """Splits ffmpeg's raw `-f mjpeg` stdout (concatenated JPEG images, no
    extra framing) back into individual frames by scanning for JPEG SOI/EOI
    markers. JPEG's byte-stuffing rule (any 0xFF inside entropy-coded scan
    data is always followed by 0x00) means 0xFFD9 cannot appear spuriously
    mid-frame, so marker-scanning is a safe way to demux this format."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> "list[bytes]":
        self._buf.extend(data)
        frames = []
        while True:
            start = self._buf.find(b"\xff\xd8")
            if start == -1:
                self._buf.clear()
                break
            end = self._buf.find(b"\xff\xd9", start + 2)
            if end == -1:
                if start > 0:
                    del self._buf[:start]
                break
            end += 2
            frames.append(bytes(self._buf[start:end]))
            del self._buf[:end]
        return frames


async def _run_ffmpeg_pipe(chunk_queue: "asyncio.Queue[bytes | None]") -> None:
    """Feeds Annex-B H264 access units from chunk_queue into an ffmpeg
    subprocess and pushes each decoded+re-encoded JPEG frame into _store.
    One instance per WebSocket connection, so a reconnect always gets a
    fresh decode context instead of trying to resume mid-stream."""
    proc = await asyncio.create_subprocess_exec(
        *_FFMPEG_ARGS,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def _pump_stdin() -> None:
        while True:
            chunk = await chunk_queue.get()
            if chunk is None:
                break
            proc.stdin.write(chunk)
            await proc.stdin.drain()
        proc.stdin.close()

    async def _pump_stdout() -> None:
        splitter = _MjpegSplitter()
        while True:
            data = await proc.stdout.read(65536)
            if not data:
                break
            for jpeg in splitter.feed(data):
                _store.put(jpeg)

    async def _log_stderr() -> None:
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            log.warning("mjpeg_bridge: ffmpeg: %s", line.decode(errors="replace").rstrip())

    try:
        await asyncio.gather(_pump_stdin(), _pump_stdout(), _log_stderr())
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


# ── HTTPS + WSS: capture page + frame ingest ────────────────────────────────

async def _ws_handler(connection) -> None:
    log.info("mjpeg_bridge: capture client connected (%s)", connection.remote_address)
    _store.set_connected(True)
    chunk_queue: "asyncio.Queue[bytes | None]" = asyncio.Queue()
    ffmpeg_task = asyncio.create_task(_run_ffmpeg_pipe(chunk_queue))
    try:
        async for message in connection:
            if isinstance(message, (bytes, bytearray)):
                chunk_queue.put_nowait(bytes(message))
    finally:
        chunk_queue.put_nowait(None)
        await ffmpeg_task
        _store.set_connected(False)
        _store.clear()
        log.info("mjpeg_bridge: capture client disconnected (%s)", connection.remote_address)


def _process_request(connection, request):
    if request.path == "/ingest":
        return None  # let the WebSocket handshake proceed
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    body = _CAPTURE_PAGE.encode("utf-8")
    headers = Headers()
    headers["Content-Type"] = "text/html; charset=utf-8"
    headers["Content-Length"] = str(len(body))
    # This page's JS has changed several times during development; without
    # this, a reloaded Safari tab can silently keep running a cached older
    # version (e.g. the old software-JPEG capture loop) and produce
    # confusing "why is it still slow" results that look like a live test.
    headers["Cache-Control"] = "no-store"
    return Response(200, "OK", headers, body)


async def _run_wss_server() -> None:
    from websockets.asyncio.server import serve

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(config.MJPEG_BRIDGE_TLS_CERT, config.MJPEG_BRIDGE_TLS_KEY)

    async with serve(
        _ws_handler,
        "0.0.0.0",
        config.MJPEG_BRIDGE_WSS_PORT,
        ssl=ssl_ctx,
        process_request=_process_request,
    ) as server:
        log.info(
            "mjpeg_bridge: HTTPS capture page + WSS ingest on port %d",
            config.MJPEG_BRIDGE_WSS_PORT,
        )
        await server.serve_forever()


# ── plain HTTP: MJPEG output + health ────────────────────────────────────────

class _HttpHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        log.debug("mjpeg_bridge http: " + fmt, *args)

    def do_GET(self) -> None:
        if self.path == "/video":
            self._serve_video()
        elif self.path == "/health":
            self._serve_health()
        else:
            self.send_error(404)

    def _serve_health(self) -> None:
        body = json.dumps(_store.status()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_video(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.end_headers()

        last_ts = 0.0
        stalls = 0
        try:
            while True:
                got = _store.wait_for_frame(last_ts, timeout=_STALL_TIMEOUT_S)
                if got is None:
                    stalls += 1
                    if stalls >= _MAX_CONSECUTIVE_STALLS:
                        log.info("mjpeg_bridge: /video closing — no frames for %.0fs", _STALL_TIMEOUT_S * stalls)
                        break
                    continue
                stalls = 0
                jpeg, last_ts = got
                chunk = (
                    f"--{_BOUNDARY}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode("ascii") + jpeg + b"\r\n"
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _run_http_server() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", config.MJPEG_BRIDGE_HTTP_PORT), _HttpHandler)
    log.info("mjpeg_bridge: plain HTTP MJPEG output on port %d", config.MJPEG_BRIDGE_HTTP_PORT)
    server.serve_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if _FFMPEG_BIN is None:
        raise RuntimeError("ffmpeg not found on PATH — required to decode the H264 capture feed")
    http_thread = threading.Thread(target=_run_http_server, daemon=True)
    http_thread.start()
    asyncio.run(_run_wss_server())


if __name__ == "__main__":
    main()
