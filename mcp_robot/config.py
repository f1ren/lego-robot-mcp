"""
Environment-based configuration for the Lego robot MCP server.

Motor port mapping must match the physical wiring on your BuildHat.
Adjust PORT_* variables (or set env vars) if the robot behaves unexpectedly.
"""
import os
from pathlib import Path

_PROJECT_DIR = Path(__file__).parent.parent
_OUTPUT_DIR = _PROJECT_DIR / "output"

# ── SSH ──────────────────────────────────────────────────────────────────────
RPI_HOST = os.getenv("ROBOT_HOST", "rpi.local")
RPI_USER = os.getenv("ROBOT_USER", "rpi")
SSH_TIMEOUT = int(os.getenv("SSH_TIMEOUT", "10"))

# ── Motor port mapping ───────────────────────────────────────────────────────
# Adjust these to match the physical BuildHat wiring.
PORT_LEFT_WHEEL  = os.getenv("PORT_LEFT_WHEEL",  "A")
PORT_RIGHT_WHEEL = os.getenv("PORT_RIGHT_WHEEL", "B")
PORT_ARM         = os.getenv("PORT_ARM",         "D")
PORT_GRIPPER     = os.getenv("PORT_GRIPPER",     "C")

# ── Gripper calibration (relative degrees) ───────────────────────────────────
# These are RELATIVE travel amounts, not absolute targets.  LEGO motors use
# incremental encoders that reset on power-cycle, so an absolute target is
# unreliable.  GRIPPER_OPEN_DEG is how far the motor turns to go from closed
# to fully open (~180° finger separation); GRIPPER_CLOSED_DEG is how far it
# turns to go from open to closed.
GRIPPER_OPEN_DEG   = int(os.getenv("GRIPPER_OPEN_DEG",   "180"))
GRIPPER_CLOSED_DEG = int(os.getenv("GRIPPER_CLOSED_DEG", "90"))

# ── Arm limits (degrees relative to motor home) ───────────────────────────────
ARM_UP_DEG   = int(os.getenv("ARM_UP_DEG",   "0"))    # home / retracted
ARM_DOWN_DEG = int(os.getenv("ARM_DOWN_DEG", "90"))   # extended / lowered

# ── Default speeds (range -100 to 100) ───────────────────────────────────────
DEFAULT_WHEEL_SPEED   = int(os.getenv("DEFAULT_WHEEL_SPEED",   "50"))
DEFAULT_ARM_SPEED     = int(os.getenv("DEFAULT_ARM_SPEED",     "30"))
DEFAULT_GRIPPER_SPEED = int(os.getenv("DEFAULT_GRIPPER_SPEED", "25"))

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_WIDTH   = int(os.getenv("CAMERA_WIDTH",   "640"))
CAMERA_HEIGHT  = int(os.getenv("CAMERA_HEIGHT",  "480"))
CAMERA_WARMUP  = float(os.getenv("CAMERA_WARMUP", "0.8"))  # seconds
POST_ACTION_SETTLE = float(os.getenv("POST_ACTION_SETTLE", "0.5"))  # settle delay before after-capture

# ── DroidCam ──────────────────────────────────────────────────────────────────
DROIDCAM_URL         = os.getenv("DROIDCAM_URL", "http://192.168.8.186:4747/video")
# Target capture rate for DroidCam during action execution and video compilation.
# Higher = smoother video and better optical flow; limited by DroidCam's native rate (~30 fps).
DROIDCAM_CAPTURE_FPS = float(os.getenv("DROIDCAM_CAPTURE_FPS", "15.0"))

# ── Pi Camera MJPEG HTTP server ───────────────────────────────────────────────
# stream_live() starts a picamera2 MJPEG server on the RPi and reads it via
# OpenCV — no per-frame SSH overhead, achieves 15-30 fps at 640×480.
PICAMERA_MJPEG_PORT = int(os.getenv("PICAMERA_MJPEG_PORT", "8765"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.getenv("LOG_FILE", str(_OUTPUT_DIR / "logs" / "mcp_server.log"))

# ── Snapshots ─────────────────────────────────────────────────────────────────
# Directory where captured images are saved before being sent to the model.
# Set SNAPSHOT_DIR="" to disable saving.
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", str(_OUTPUT_DIR / "snapshots"))

# ── Vision backend ────────────────────────────────────────────────────────────
# VISION_BACKEND: "gemini" | "ollama" | "auto"
#   auto = try Gemini first, fall back to Ollama on failure/quota exhaustion
VISION_BACKEND = os.getenv("VISION_BACKEND", "auto")

# ── Gemini vision (Robotics-ER) ──────────────────────────────────────────────
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL          = os.getenv("GEMINI_MODEL",          "gemini-robotics-er-1.6-preview")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-pro")

# ── Ollama local vision ───────────────────────────────────────────────────────
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:32b-thinking")

# ── Rerun visualization (optional) ───────────────────────────────────────────
# RERUN_ENABLED=1          enable rerun logging
# RERUN_MODE=spawn         launch the desktop viewer (default)
# RERUN_MODE=serve         serve gRPC + web viewer
# RERUN_CONNECT=1          connect to an already-running viewer (use in MCP server
#                          when stream.py has already spawned the viewer)
# RERUN_ADDR               gRPC URL for RERUN_CONNECT (default: rerun+http://127.0.0.1:9876)
RERUN_ENABLED = bool(os.getenv("RERUN_ENABLED", ""))
RERUN_MODE    = os.getenv("RERUN_MODE", "spawn")
RERUN_CONNECT = bool(os.getenv("RERUN_CONNECT", ""))
RERUN_ADDR    = os.getenv("RERUN_ADDR", "rerun+http://127.0.0.1:9876")
