#!/home/navatm/Projects/lego-robot-mcp/.venv/bin/python3
"""
Compile a task video by concatenating recorded motion segments since a given
timestamp.

Usage:
    ./compile_video.py "2026-05-07 12:16:41,594"
    ./compile_video.py 20260507_121641
    ./compile_video.py 1746613200.0
    ./compile_video.py 1746613200.0 --camera pi_camera
"""
import argparse
import logging

from mcp_robot import config
from mcp_robot.video_compiler import compile_task_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Compile a task video from recorded motion segments.")
    parser.add_argument("since", help="Start timestamp (log format, folder format, or UNIX float)")
    parser.add_argument("--camera", choices=["droidcam", "pi_camera"], default="droidcam",
                        help="Camera whose segments to compile (default: droidcam)")
    args = parser.parse_args()

    result = compile_task_video(args.since, config.SEGMENT_MANIFEST, args.camera)
    if not result.ok:
        logging.error(result.error)
        return 1
    if result.video_path is None:
        logging.warning("No segments found since %s for camera %s", args.since, args.camera)
        return 0
    print(result.video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
