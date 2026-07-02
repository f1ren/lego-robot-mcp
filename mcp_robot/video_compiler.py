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

from mcp_robot import config

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


# ── merged (tiled) task video ────────────────────────────────────────────────
# Tiles both cameras plus a subtitle card into a single video:
#   left-top    = subtitle (observation, then action) on black
#   left-bottom = front (Pi) camera
#   right       = external (DroidCam) camera, full column height
#
# The two cameras record independent, motion-triggered segments (see
# recorder.py) that are only approximately aligned in time (cross-triggering
# keeps them within SEGMENT_CROSS_TRIGGER_WINDOW of each other). To produce
# one coherent video we first merge same-time segments from both cameras
# into "scenes" (_group_into_scenes), then render each scene as a tile
# composite of matched duration, then concatenate the scenes.


def _probe_resolution(path: str) -> tuple[int, int] | None:
    """Return (width, height) of a video file's first stream, or None."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        w_str, h_str = proc.stdout.strip().split("x")
        return int(w_str), int(h_str)
    except Exception as exc:
        log.warning("video_compiler: ffprobe failed for %s: %s", path, exc)
        return None


def _even(n: float) -> int:
    """Round to the nearest integer, then up to even (yuv420p needs even dims)."""
    n = int(round(n))
    return n if n % 2 == 0 else n + 1


def _first_probe(segments: list[dict]) -> tuple[int, int] | None:
    """Return the native (width, height) of the first probeable segment."""
    for seg in segments:
        res = _probe_resolution(seg["path"])
        if res:
            return res
    return None


def _group_into_scenes(segments: list[dict], gap_s: float) -> list[list[dict]]:
    """Merge segments (from either camera, pre-sorted by start_ts) into
    scenes: a segment joins the running scene if it starts within `gap_s` of
    the scene's current end, else it starts a new scene. Cross-camera
    triggering keeps both cameras' segments for the same action close in
    time, so this reunites them without relying on tag text (which need not
    be unique across actions)."""
    scenes: list[list[dict]] = []
    scene_end = None
    for seg in segments:
        if scenes and seg["start_ts"] <= scene_end + gap_s:
            scenes[-1].append(seg)
            scene_end = max(scene_end, seg["end_ts"])
        else:
            scenes.append([seg])
            scene_end = seg["end_ts"]
    return scenes


def _scene_subtitle(scene: list[dict]) -> tuple[str, str]:
    """Return (observation, action) text for a scene: the first non-empty
    tag among its segments (both cameras' segments for the same action carry
    identical tags — see recorder.tag_range)."""
    for seg in scene:
        obs, act = seg.get("sub_observation"), seg.get("sub_action")
        if obs or act:
            return obs or "", act or ""
    return "", ""


_SUB_OBS_COLOR_BGR = (255, 255, 255)   # white — observation line
_SUB_ACT_COLOR_BGR = (0, 165, 255)     # orange — action line (matches heading._OBJ_COLOR_BGR)


def _write_subtitle_image(path: str, width: int, height: int, observation: str, action: str) -> None:
    """Render a black width×height frame with the observation (white)
    stacked above the action (orange), auto-shrunk to fit, and save as PNG."""
    import cv2
    import numpy as np

    img = np.zeros((height, width, 3), dtype=np.uint8)
    lines = [(t.strip(), c) for t, c in
             ((observation, _SUB_OBS_COLOR_BGR), (action, _SUB_ACT_COLOR_BGR)) if t and t.strip()]
    if not lines:
        cv2.imwrite(path, img)
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, height // 200)
    margin = max(8, width // 30)
    max_w = width - 2 * margin

    scale = max(0.4, height / 260.0)
    for text, _color in lines:
        while scale > 0.3:
            (tw, _th), _base = cv2.getTextSize(text, font, scale, thickness)
            if tw <= max_w:
                break
            scale -= 0.05

    (_, th), baseline = cv2.getTextSize("Ag", font, scale, thickness)
    line_gap = th + baseline + max(6, height // 20)
    y = (height - line_gap * len(lines)) // 2 + th
    for text, color in lines:
        cv2.putText(img, text, (margin, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_gap

    cv2.imwrite(path, img)


def _scale_pad(label_in: str, w: int, h: int, label_out: str, fps: float) -> str:
    """Fit-scale + letterbox-pad a stream to exactly w×h, normalize fps/format."""
    return (f"[{label_in}]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
            f"fps={fps},format=yuv420p[{label_out}]")


def _pad_duration(label_in: str, dur: float, label_out: str) -> str:
    """Freeze-pad (if short) and hard-trim a stream to exactly `dur` seconds."""
    return (f"[{label_in}]tpad=stop_mode=clone:stop_duration={dur:.3f},"
            f"trim=duration={dur:.3f},setpts=PTS-STARTPTS[{label_out}]")


def _append_camera_branch(
    inputs: list[str], filters: list[str], next_idx: int, segs: list[dict],
    w: int, h: int, dur: float, fps: float, label_out: str,
) -> int:
    """Append ffmpeg -i args and filter-graph fragments (in place) that
    produce a single [label_out] stream of exactly `dur` seconds at w×h for
    one camera's footage within a scene. Falls back to a black filler when
    the camera recorded nothing in this window; each segment is scaled
    individually before concatenation (needed if more than one segment
    falls in the same scene, since concat requires matching dimensions)."""
    if not segs:
        inputs += ["-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:d={dur:.3f}:r={fps}"]
        raw_labels = [f"{next_idx}:v"]
        next_idx += 1
    else:
        raw_labels = []
        for seg in segs:
            inputs += ["-i", seg["path"]]
            raw_labels.append(f"{next_idx}:v")
            next_idx += 1

    scaled_labels = []
    for j, raw_label in enumerate(raw_labels):
        scaled_label = f"{label_out}s{j}"
        filters.append(_scale_pad(raw_label, w, h, scaled_label, fps))
        scaled_labels.append(scaled_label)

    if len(scaled_labels) == 1:
        merged_label = scaled_labels[0]
    else:
        merged_label = f"{label_out}cat"
        filters.append("".join(f"[{l}]" for l in scaled_labels) +
                        f"concat=n={len(scaled_labels)}:v=1:a=0[{merged_label}]")

    filters.append(_pad_duration(merged_label, dur, label_out))
    return next_idx


def _render_scene(
    droid_segs: list[dict], pi_segs: list[dict], subtitle_png: str, dur: float,
    right_w: int, right_h: int, cell_w: int, cell_h: int, fps: float, out_path: str,
) -> str | None:
    """Render one tiled scene clip (subtitle top-left, Pi camera bottom-left,
    DroidCam full-height right). Returns an error string, or None on success."""
    inputs: list[str] = []
    filters: list[str] = []
    next_idx = 0

    next_idx = _append_camera_branch(inputs, filters, next_idx, droid_segs, right_w, right_h, dur, fps, "ext")
    next_idx = _append_camera_branch(inputs, filters, next_idx, pi_segs, cell_w, cell_h, dur, fps, "pic")

    inputs += ["-loop", "1", "-t", f"{dur:.3f}", "-r", f"{fps}", "-i", subtitle_png]
    sub_raw = f"{next_idx}:v"
    next_idx += 1
    filters.append(_scale_pad(sub_raw, cell_w, cell_h, "subs", fps))
    filters.append(_pad_duration("subs", dur, "sub"))

    filters.append("[sub][pic]vstack=inputs=2[left]")
    filters.append("[left][ext]hstack=inputs=2[outv]")

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[outv]", "-r", f"{fps}", "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return proc.stderr.strip()[-2000:]
    return None


def _read_segments_by_camera(manifest_path: str, since_ts: float, cameras: tuple[str, ...]) -> dict[str, list[dict]]:
    by_camera: dict[str, list[dict]] = {cam: [] for cam in cameras}
    with open(manifest_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cam = rec.get("camera")
            if cam not in by_camera:
                continue
            if rec.get("end_ts") is None:
                continue
            if rec.get("start_ts", -1) < since_ts:
                continue
            if rec.get("frame_count", 0) <= 0:
                continue
            if not os.path.isfile(rec.get("path", "")):
                continue
            by_camera[cam].append(rec)
    for segs in by_camera.values():
        segs.sort(key=lambda r: r["start_ts"])
    return by_camera


def compile_merged_video(
    since: str,
    manifest_path: str,
    out_dir: str = "~/Videos/LegoRobot",
) -> CompileResult:
    """Compile a single video tiling both cameras plus a subtitle card since
    `since` (see module docstring above for the layout and scene-merging
    strategy)."""
    try:
        since_ts = parse_since(since)
    except ValueError as exc:
        return CompileResult(None, 0, 0.0, error=str(exc))

    if not os.path.isfile(manifest_path):
        return CompileResult(None, 0, 0.0)

    try:
        by_camera = _read_segments_by_camera(manifest_path, since_ts, ("droidcam", "pi_camera"))
    except Exception as exc:
        return CompileResult(None, 0, 0.0, error=f"Failed to read manifest: {exc}")

    all_segments = sorted(by_camera["droidcam"] + by_camera["pi_camera"], key=lambda r: r["start_ts"])
    if not all_segments:
        return CompileResult(None, 0, 0.0)

    scenes = _group_into_scenes(all_segments, config.MERGE_SCENE_GAP_S)

    cell_h = _even(config.MERGE_CELL_HEIGHT)
    pi_w0, pi_h0 = _first_probe(by_camera["pi_camera"]) or (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    cell_w = _even(cell_h * pi_w0 / pi_h0)

    right_h = cell_h * 2
    droid_native = _first_probe(by_camera["droidcam"])
    if droid_native:
        droid_w0, droid_h0 = droid_native
    elif config.DROIDCAM_ROTATION in (90, 270):
        droid_w0, droid_h0 = config.CAMERA_HEIGHT, config.CAMERA_WIDTH
    else:
        droid_w0, droid_h0 = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
    right_w = _even(right_h * droid_w0 / droid_h0)

    fps = config.MERGE_FPS
    out_dir_expanded = os.path.expanduser(out_dir)
    os.makedirs(out_dir_expanded, exist_ok=True)
    out_path = os.path.join(out_dir_expanded, f"task_video_merged_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

    total_duration_s = 0.0
    with tempfile.TemporaryDirectory(dir=out_dir_expanded) as tmp_dir:
        scene_paths = []
        for i, scene in enumerate(scenes):
            t0 = min(s["start_ts"] for s in scene)
            t1 = max(s["end_ts"] for s in scene)
            dur = max(t1 - t0, 1.0 / fps)
            droid_segs = [s for s in scene if s["camera"] == "droidcam"]
            pi_segs = [s for s in scene if s["camera"] == "pi_camera"]
            observation, action = _scene_subtitle(scene)

            log.info("Scene %d: t0=%.3f dur=%.1fs droid_segs=%d pi_segs=%d obs=%r act=%r",
                      i, t0, dur, len(droid_segs), len(pi_segs), observation, action)

            sub_png = os.path.join(tmp_dir, f"sub_{i:04d}.png")
            _write_subtitle_image(sub_png, cell_w, cell_h, observation, action)

            scene_out = os.path.join(tmp_dir, f"scene_{i:04d}.mp4")
            err = _render_scene(droid_segs, pi_segs, sub_png, dur,
                                 right_w, right_h, cell_w, cell_h, fps, scene_out)
            if err:
                return CompileResult(None, 0, 0.0, error=f"Scene {i} render failed: {err}")
            scene_paths.append(scene_out)
            total_duration_s += dur

        list_path = os.path.join(tmp_dir, "concat_list.txt")
        with open(list_path, "w") as fh:
            for p in scene_paths:
                escaped = os.path.abspath(p).replace("'", "'\\''")
                fh.write(f"file '{escaped}'\n")

        proc = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return CompileResult(None, 0, 0.0, error=proc.stderr.strip()[-2000:])

    log.info("Merged task video written: %s (%d scene(s) from %d segment(s), %.1fs total)",
              out_path, len(scenes), len(all_segments), total_duration_s)
    return CompileResult(out_path, len(all_segments), total_duration_s)
