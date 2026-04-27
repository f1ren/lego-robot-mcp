"""
Standalone preview: camera stream + motor positions in Rerun viewer.

No MCP server needed — runs directly against the RPi over SSH.

Usage:
    .venv/bin/python3 -m mcp_robot.preview
    .venv/bin/python3 -m mcp_robot.preview --fps 3 --motor-hz 1
    RERUN_MODE=serve .venv/bin/python3 -m mcp_robot.preview

Ctrl-C to stop.
"""
import argparse
import logging
import threading

from mcp_robot import camera, config, robot, viz

log = logging.getLogger(__name__)

# This script's only job is visualization — force Rerun on.
config.RERUN_ENABLED = True


def _poll_motors(stop: threading.Event, interval: float) -> None:
    while not stop.is_set():
        try:
            robot.get_all_positions()
        except Exception as exc:
            log.warning("Motor poll failed: %s", exc)
        stop.wait(interval)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Camera + motor preview in Rerun")
    parser.add_argument("--fps", type=float, default=5.0, help="Camera frame rate (default 5)")
    parser.add_argument("--motor-hz", type=float, default=2.0, help="Motor poll rate in Hz (default 2)")
    args = parser.parse_args()

    stop = threading.Event()

    # Seed one motor read so Rerun initializes before the camera stream starts.
    log.info("Initializing motors...")
    try:
        robot.get_all_positions()
        log.info("Motors ready.")
    except Exception as exc:
        log.warning("Motor init failed (%s) — continuing without motor data.", exc)

    motor_thread = threading.Thread(
        target=_poll_motors, args=(stop, 1.0 / args.motor_hz), daemon=True,
    )
    motor_thread.start()

    def _run_droidcam() -> None:
        try:
            camera.stream_droidcam(stop_event=stop)
        except Exception as exc:
            log.error("DroidCam stream failed: %s", exc)

    droidcam_thread = threading.Thread(target=_run_droidcam, daemon=True)
    droidcam_thread.start()

    log.info("Camera %.0f fps  |  Motors %.0f Hz  —  Ctrl-C to stop", args.fps, args.motor_hz)
    try:
        camera.stream_live(fps=args.fps, stop_event=stop)
    except KeyboardInterrupt:
        log.info("Interrupted.")
    except RuntimeError as exc:
        log.error("Stream error: %s", exc)
        raise SystemExit(1)
    finally:
        stop.set()
        motor_thread.join(timeout=2)
        droidcam_thread.join(timeout=2)

    viz.flush()
    log.info("Done.")


if __name__ == "__main__":
    main()
