"""
Live camera stream → rerun viewer.

Usage:
    RERUN_ENABLED=1 .venv/bin/python3 -m mcp_robot.stream
    RERUN_ENABLED=1 .venv/bin/python3 -m mcp_robot.stream --fps 3
    RERUN_ENABLED=1 RERUN_MODE=serve .venv/bin/python3 -m mcp_robot.stream

The rerun binary in the venv (.venv/bin/rerun) is used automatically.
Ctrl-C to stop.
"""
import argparse
import signal
import threading

from mcp_robot import camera, viz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=5.0, help="Target frames per second")
    args = parser.parse_args()

    stop = threading.Event()
    signal.signal(signal.SIGINT,  lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    print(f"Streaming at {args.fps} fps — Ctrl-C to stop")
    try:
        camera.stream_live(fps=args.fps, stop_event=stop)
    except RuntimeError as exc:
        print(f"Stream error: {exc}")
        raise SystemExit(1)
    viz.flush()
    print("Stream ended.")


if __name__ == "__main__":
    main()
