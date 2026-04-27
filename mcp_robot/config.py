"""
Environment-based configuration for the Lego robot MCP server.

Motor port mapping must match the physical wiring on your BuildHat.
Adjust PORT_* variables (or set env vars) if the robot behaves unexpectedly.
"""
import os

# ── SSH ──────────────────────────────────────────────────────────────────────
RPI_HOST = os.getenv("ROBOT_HOST", "rpi.local")
RPI_USER = os.getenv("ROBOT_USER", "rpi")
SSH_TIMEOUT = int(os.getenv("SSH_TIMEOUT", "10"))

# ── Motor port mapping ───────────────────────────────────────────────────────
# Adjust these to match the physical BuildHat wiring.
PORT_LEFT_WHEEL  = os.getenv("PORT_LEFT_WHEEL",  "A")
PORT_RIGHT_WHEEL = os.getenv("PORT_RIGHT_WHEEL", "B")
PORT_ARM         = os.getenv("PORT_ARM",         "C")
PORT_GRIPPER     = os.getenv("PORT_GRIPPER",     "D")

# ── Gripper calibration (degrees) ────────────────────────────────────────────
# Run control_gripper manually and observe positions to calibrate.
GRIPPER_OPEN_DEG   = int(os.getenv("GRIPPER_OPEN_DEG",   "0"))
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

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL",   "gemini-1.5-flash")

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
