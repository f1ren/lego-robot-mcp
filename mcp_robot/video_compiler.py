"""Compile task videos by concatenating tagged motion segments from the
SegmentRecorder manifest (see mcp_robot/recorder.py)."""
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime

log = logging.getLogger(__name__)


@dataclass
class CompileResult:
    video_path: str | None
    segment_count: int
    total_duration_s: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_since(since: str) -> float:
    """Parse a UNIX-timestamp string, or a legacy datetime/folder-name format,
    into a UNIX float timestamp."""
    since = since.strip()
    try:
        return float(since)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(since, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse 'since' timestamp: {since!r}")


def compile_task_video(
    since: str,
    manifest_path: str,
    camera: str = "droidcam",
    out_dir: str = "~/Videos/LegoRobot",
) -> CompileResult:
    try:
        since_ts = parse_since(since)
    except ValueError as exc:
        return CompileResult(None, 0, 0.0, error=str(exc))

    if not os.path.isfile(manifest_path):
        return CompileResult(None, 0, 0.0)

    segments = []
    try:
        with open(manifest_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("camera") != camera:
                    continue
                if rec.get("end_ts") is None:
                    continue
                if rec.get("start_ts", -1) < since_ts:
                    continue
                if rec.get("frame_count", 0) <= 0:
                    continue
                if not os.path.isfile(rec.get("path", "")):
                    continue
                segments.append(rec)
    except Exception as exc:
        return CompileResult(None, 0, 0.0, error=f"Failed to read manifest: {exc}")

    if not segments:
        return CompileResult(None, 0, 0.0)

    segments.sort(key=lambda r: r["start_ts"])

    for rec in segments:
        log.info("Segment: %s (start=%.3f, dur=%.1fs, frames=%d, tool=%s)",
                 rec["path"], rec["start_ts"], rec["end_ts"] - rec["start_ts"],
                 rec.get("frame_count", 0), rec.get("tool"))

    out_dir_expanded = os.path.expanduser(out_dir)
    os.makedirs(out_dir_expanded, exist_ok=True)
    out_path = os.path.join(out_dir_expanded, f"task_video_{camera}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", dir=out_dir_expanded, delete=False) as fh:
        list_path = fh.name
        for rec in segments:
            abspath = os.path.abspath(rec["path"])
            escaped = abspath.replace("'", "'\\''")
            fh.write(f"file '{escaped}'\n")

    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path],
            capture_output=True, text=True,
        )
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass

    if proc.returncode != 0:
        return CompileResult(None, 0, 0.0, error=proc.stderr.strip()[-2000:])

    total_duration_s = sum(r["end_ts"] - r["start_ts"] for r in segments)
    log.info("Task video written: %s (%d segment(s), %.1fs total)", out_path, len(segments), total_duration_s)
    return CompileResult(out_path, len(segments), total_duration_s)
