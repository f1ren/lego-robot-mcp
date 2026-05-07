#!/home/navatm/Projects/lego-robot-mcp/.venv/bin/python3
"""
Compile a task video from action_video_* snapshot folders.

Usage:
    ./compile_video.py "2026-05-07 12:16:41,594"
    ./compile_video.py 20260507_121641
    ./compile_video.py 1746613200.0
    ./compile_video.py "2026-05-07 12:16:41,594" --fps 10
"""
import argparse
import logging
import os

from mcp_robot.video_compiler import compile_task_video

SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "/tmp/lego-robot-snapshots")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Compile a task video from snapshot folders.")
    parser.add_argument("since", help="Start timestamp (log format, folder format, or UNIX float)")
    parser.add_argument("--fps", type=float, default=5.0, help="Output playback FPS (default 5.0)")
    args = parser.parse_args()

    result = compile_task_video(args.since, SNAPSHOT_DIR, args.fps)
    if not result.ok:
        logging.error(result.error)
        return 1
    if result.video_path is None:
        logging.warning("No motion detected — video not written (%d frames, %d actions)",
                        result.total_frames, result.action_count)
        return 0
    print(result.video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
