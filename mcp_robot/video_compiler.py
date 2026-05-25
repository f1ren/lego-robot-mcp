"""Core logic for compiling task videos from action_video_* snapshot folders."""
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

MOTION_THRESHOLD = 2.0  # mean absolute pixel diff (0-255 scale)

log = logging.getLogger(__name__)


@dataclass
class CompileResult:
    video_path: str | None
    total_frames: int
    motion_frames: int
    action_count: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_since(since: str) -> str:
    """Convert any supported timestamp format to YYYYMMDD_HHMMSS for folder comparison."""
    since = since.strip()
    try:
        return time.strftime("%Y%m%d_%H%M%S", time.localtime(float(since)))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(since, fmt).strftime("%Y%m%d_%H%M%S")
        except ValueError:
            continue
    return since  # already in folder-name format or unknown


def compile_task_video(
    since: str,
    snapshot_dir: str,
    output_fps: float = 5.0,
    out_dir: str = "~/Videos/LegoRobot",
) -> CompileResult:
    since_dt = parse_since(since)
    log.info("compile_video: since_dt=%r (parsed from %r)", since_dt, since)

    if not snapshot_dir or not os.path.isdir(snapshot_dir):
        return CompileResult(None, 0, 0, 0, error=f"SNAPSHOT_DIR not found: {snapshot_dir!r}")

    try:
        action_dirs = sorted([
            os.path.join(snapshot_dir, d)
            for d in os.listdir(snapshot_dir)
            if d.startswith("action_video_")
            and os.path.isdir(os.path.join(snapshot_dir, d))
            and d[len("action_video_"):] >= since_dt
        ])
    except Exception as exc:
        return CompileResult(None, 0, 0, 0, error=f"Failed to list SNAPSHOT_DIR: {exc}")

    log.info("compile_video: %d folder(s) filtered in: %s", len(action_dirs), action_dirs)

    if not action_dirs:
        return CompileResult(None, 0, 0, 0,
                             error=f"No action_video_* folders found since {since_dt!r} in {snapshot_dir}")

    all_paths: list[str] = []
    for action_dir in action_dirs:
        try:
            all_paths.extend(sorted(
                os.path.join(action_dir, f)
                for f in os.listdir(action_dir)
                if f.endswith(".jpg") and "droidcam" in f
            ))
        except Exception as exc:
            log.warning("Skipping action dir %s: %s", action_dir, exc)

    if not all_paths:
        return CompileResult(None, 0, 0, len(action_dirs), error="No JPEG frames found in matched folders")

    imgs = [(fp, cv2.imread(fp)) for fp in all_paths]
    imgs = [(fp, img) for fp, img in imgs if img is not None]
    if not imgs:
        return CompileResult(None, 0, 0, len(action_dirs), error="Failed to decode any frames")

    n = len(imgs)
    in_motion = [False] * n
    for i in range(n - 1):
        try:
            gray_a = cv2.cvtColor(imgs[i][1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_b = cv2.cvtColor(imgs[i + 1][1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = float(np.mean(np.abs(gray_a - gray_b)))
        except Exception as exc:
            raise RuntimeError(f"Failed to process frames: {imgs[i][0]!r} and {imgs[i + 1][0]!r}") from exc
        if diff > MOTION_THRESHOLD:
            in_motion[i] = True
            in_motion[i + 1] = True

    motion_imgs = [(imgs[i][0], imgs[i][1]) for i, flag in enumerate(in_motion) if flag]
    log.info("compile_video: %d/%d motion frames", len(motion_imgs), n)

    if not motion_imgs:
        return CompileResult(None, n, 0, len(action_dirs))

    h, w = motion_imgs[0][1].shape[:2]
    out_dir_expanded = os.path.expanduser(out_dir)
    os.makedirs(out_dir_expanded, exist_ok=True)
    video_path = os.path.join(out_dir_expanded, f"task_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (w, h))
    for fp, img in motion_imgs:
        try:
            writer.write(img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h)))
        except Exception as exc:
            raise RuntimeError(f"Failed to write frame: {fp!r}") from exc
    writer.release()

    log.info("Task video written: %s (%d/%d motion frames from %d actions since %s)",
             video_path, len(motion_imgs), n, len(action_dirs), since_dt)
    return CompileResult(video_path, n, len(motion_imgs), len(action_dirs))
