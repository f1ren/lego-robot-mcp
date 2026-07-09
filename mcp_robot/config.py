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

# ── Wheel geometry (for body-angle → encoder-degree conversion) ──────────────
# Measure on the actual robot.  All values in mm.
# encoder_deg_per_body_deg = TRACK_WIDTH_MM / WHEEL_DIAMETER_MM
# Override TURN_ENCODER_DEG_PER_BODY_DEG directly to skip the geometry math.
WHEEL_DIAMETER_MM = float(os.getenv("WHEEL_DIAMETER_MM", "56.0"))
TRACK_WIDTH_MM    = float(os.getenv("TRACK_WIDTH_MM",    "123.0"))
TURN_ENCODER_DEG_PER_BODY_DEG = float(
    os.getenv("TURN_ENCODER_DEG_PER_BODY_DEG",
              str(TRACK_WIDTH_MM / WHEEL_DIAMETER_MM))
)

# ── Robot chassis footprint (for pixel → mm distance calibration) ────────────
# Visible yellow top surface, measured with a ruler: 152 x 88 mm
# (= 19 x 11 LEGO studs at 8mm/stud).  Navigation uses the *area* — rather
# than a single length — to convert pixel distances to real-world mm, because
# area is rotation-invariant: an axis-aligned bounding-box measurement (e.g.
# robot_radius_px) changes with the robot's heading, but the visible yellow
# pixel count does not.
ROBOT_BODY_LENGTH_MM = float(os.getenv("ROBOT_BODY_LENGTH_MM", "152.0"))
ROBOT_BODY_WIDTH_MM  = float(os.getenv("ROBOT_BODY_WIDTH_MM",  "88.0"))
ROBOT_BODY_AREA_MM2  = ROBOT_BODY_LENGTH_MM * ROBOT_BODY_WIDTH_MM

# ── Speed limits ─────────────────────────────────────────────────────────────
# Minimum absolute speed for any motor action. Below this threshold motion is
# too slow for the CV pipeline to reliably detect, making success verification
# impossible. Wheel speeds may also be 0 (coast / pivot one wheel).
SPEED_MIN = int(os.getenv("SPEED_MIN", "15"))
SPEED_MAX = int(os.getenv("SPEED_MAX", "20"))

# ── Default speeds (range -100 to 100) ───────────────────────────────────────
DEFAULT_WHEEL_SPEED   = int(os.getenv("DEFAULT_WHEEL_SPEED",   "50"))
# Halved from 15: lift_arm/lower_arm raise (or lower) and then fall back
# afterward; running slower makes the raise-then-fall easier to observe and
# diagnose. See ARM_SPEED_MIN below — it had to move down with this default,
# since it used to sit exactly at the shared SPEED_MIN floor (15).
DEFAULT_ARM_SPEED     = int(os.getenv("DEFAULT_ARM_SPEED",     "7"))
DEFAULT_GRIPPER_SPEED = int(os.getenv("DEFAULT_GRIPPER_SPEED", "25"))

# Arm motor speed cap — slower to avoid jitter, especially with a load.
ARM_SPEED_MAX = int(os.getenv("ARM_SPEED_MAX", "15"))
# Arm-specific floor, decoupled from the shared SPEED_MIN (which stays 15 for
# wheels/gripper, per the CV-detectability rule above) — DEFAULT_ARM_SPEED is
# now below that shared floor, so arm moves need their own.
ARM_SPEED_MIN = int(os.getenv("ARM_SPEED_MIN", "7"))

# Per-wheel speed used for in-place turns during navigate_to.
# Each wheel runs at this value (opposing directions), so effective combined
# effort is 2×NAV_TURN_SPEED. Lower values reduce overshoot and jitter.
NAV_TURN_SPEED = int(os.getenv("NAV_TURN_SPEED", "8"))

# ── Button press (click_button pre-alignment + press distance) ───────────────
# Max heading error (degrees) tolerated before click_button issues a turn
# correction to square the robot up perpendicular to the switch/button.
CLICK_ALIGN_TOLERANCE_DEG = float(os.getenv("CLICK_ALIGN_TOLERANCE_DEG", "4.0"))

# click_button drives forward by a distance derived from the measured
# distance to the switch (same px->mm calibration navigate_to uses — see
# navigation.mm_per_px), not a fixed blind duration. The measured distance is
# clamped to [CLICK_PRESS_MIN_MM, CLICK_PRESS_MAX_MM] to protect against a bad
# reading, then CLICK_PRESS_MARGIN_MM of overtravel is added — overshooting
# into a spring-loaded switch is harmless, undershooting and missing it
# entirely is the real failure mode. CLICK_PRESS_FALLBACK_MM is used only
# when the distance can't be measured at all.
CLICK_PRESS_MIN_MM      = float(os.getenv("CLICK_PRESS_MIN_MM",      "20.0"))
CLICK_PRESS_MAX_MM      = float(os.getenv("CLICK_PRESS_MAX_MM",      "200.0"))
CLICK_PRESS_MARGIN_MM   = float(os.getenv("CLICK_PRESS_MARGIN_MM",   "40.0"))
CLICK_PRESS_FALLBACK_MM = float(os.getenv("CLICK_PRESS_FALLBACK_MM", "80.0"))

# After pressing, click_button only needs to back off far enough to clear
# the switch — it does not need to return to the pre-press position. Default
# release distance is this fraction of press_degrees.
CLICK_RELEASE_FRACTION = float(os.getenv("CLICK_RELEASE_FRACTION", "0.5"))

# ── Grasp readiness (check_grasp_readiness touch gate) ────────────────────────
# Real-world gap (mm) between the target object's nearest point and the
# robot's front anchor for check_grasp_readiness() to consider it "touching"
# the body and safe to close the gripper on. Calibrated in mm — via the same
# body-plate px->mm scale drive_to()/click_button() use (navigation.mm_per_px)
# — rather than a fixed fraction of the image diagonal, because that pixel
# heuristic is not perspective-invariant: an object higher in frame (farther
# from the external camera) projects a smaller pixel gap for the same
# real-world distance, so a fixed px threshold under-detects real gaps. A
# ~112mm real gap (object clearly not touching) measured at only ~112px
# against a ~117px diagonal-fraction threshold — a false "ready" — is what
# exposed this.
GRASP_TOUCH_THRESHOLD_MM = float(os.getenv("GRASP_TOUCH_THRESHOLD_MM", "40.0"))

# ── turn_to (target-aware heading correction) ─────────────────────────────────
# Max heading error (degrees) tolerated before turn_to() issues a turn
# correction; below this it treats the robot as already facing the target and
# skips the motor command. Same default as CLICK_ALIGN_TOLERANCE_DEG but kept
# as a separate knob so tuning one doesn't silently affect the other.
TURN_TO_TOLERANCE_DEG = float(os.getenv("TURN_TO_TOLERANCE_DEG", "4.0"))

# ── Target scan (front-camera sweep when external camera can't see target) ───
SCAN_STEP_DEG  = int(os.getenv("SCAN_STEP_DEG",  "30"))   # degrees per rotation step
SCAN_TOTAL_DEG = int(os.getenv("SCAN_TOTAL_DEG", "360"))   # total arc to sweep
SCAN_SPEED     = int(os.getenv("SCAN_SPEED",     "15"))    # wheel speed during scan turns
# Set SCAN_ENABLED=1 to enable the 360° sweep; disabled by default.
SCAN_ENABLED   = bool(os.getenv("SCAN_ENABLED", ""))

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_WIDTH   = int(os.getenv("CAMERA_WIDTH",   "640"))
CAMERA_HEIGHT  = int(os.getenv("CAMERA_HEIGHT",  "480"))
CAMERA_WARMUP  = float(os.getenv("CAMERA_WARMUP", "0.8"))  # seconds
POST_ACTION_SETTLE = float(os.getenv("POST_ACTION_SETTLE", "0.5"))  # settle delay before after-capture

# ── DroidCam ──────────────────────────────────────────────────────────────────
DROIDCAM_URL         = os.getenv("DROIDCAM_URL", "http://192.168.8.190:4747/video")
DROIDCAM_ROTATION    = int(os.getenv("DROIDCAM_ROTATION", "90"))  # CW degrees: 0, 90, 180, 270
# Target capture rate for DroidCam during action execution and video compilation.
# Set to DroidCam's native ceiling (~30 fps) — the per-action VQA cost is unaffected
# by this value (vision._subsample_frames always caps to 3 frames/camera before the
# Gemini call regardless of how many were captured), so there's no downside to maxing
# it out; this only affects recorded-segment/merged-video smoothness.
DROIDCAM_CAPTURE_FPS = float(os.getenv("DROIDCAM_CAPTURE_FPS", "30.0"))

# ── Pi Camera MJPEG HTTP server ───────────────────────────────────────────────
# stream_live() starts a picamera2 MJPEG server on the RPi and reads it via
# OpenCV — no per-frame SSH overhead, achieves 15-30 fps at 640×480.
PICAMERA_MJPEG_PORT  = int(os.getenv("PICAMERA_MJPEG_PORT",  "8765"))
# Target capture rate for Pi Camera during action execution.
PICAMERA_CAPTURE_FPS = float(os.getenv("PICAMERA_CAPTURE_FPS", "5.0"))

# ── Motion segment recording ─────────────────────────────────────────────────
SEGMENT_DIR      = os.getenv("SEGMENT_DIR", str(_OUTPUT_DIR / "segments"))
SEGMENT_MANIFEST = os.getenv("SEGMENT_MANIFEST", str(Path(SEGMENT_DIR) / "index.jsonl"))
SEGMENT_PREROLL_S  = float(os.getenv("SEGMENT_PREROLL_S",  "0.25"))
SEGMENT_COOLDOWN_S = float(os.getenv("SEGMENT_COOLDOWN_S", "0.25"))
# Declared playback fps per camera (container metadata for cv2.VideoWriter).
# DroidCam: matches DROIDCAM_CAPTURE_FPS (per-action throttle rate). Keep these two
# in sync — a mismatch still self-corrects via _maybe_remux's itsscale retiming, but
# every segment then pays for an avoidable ffmpeg remux subprocess on close.
SEGMENT_FPS_DROIDCAM = float(os.getenv("SEGMENT_FPS_DROIDCAM", "30.0"))
# Pi camera: NOT PICAMERA_CAPTURE_FPS (5.0) — that low rate is the bottleneck
# this feature is meant to move past. The recorder only ever sees pi_camera
# frames when the MJPEG continuous stream is active (15-30fps native), so
# declare 15 as the floor of that native range.
SEGMENT_FPS_PI = float(os.getenv("SEGMENT_FPS_PI", "15.0"))
# avc1/H264 fail to open with this OpenCV/ffmpeg build (no h264 encoder
# available) — verified empirically. mp4v is the only codec that opens
# successfully and concatenates cleanly via `ffmpeg -c copy`.
SEGMENT_FOURCC = os.getenv("SEGMENT_FOURCC", "mp4v")
SEGMENT_RECENT_RING = int(os.getenv("SEGMENT_RECENT_RING", "8"))
# Per-camera noise-floor calibration: sample the first N stable frames to set
# a per-camera motion threshold (mean_diff + SIGMA * std_diff above noise).
# Set SEGMENT_CALIB_ENABLED=0 to use the fixed global threshold instead.
SEGMENT_CALIB_ENABLED = bool(int(os.getenv("SEGMENT_CALIB_ENABLED", "1")))
SEGMENT_CALIB_FRAMES  = int(os.getenv("SEGMENT_CALIB_FRAMES", "40"))
SEGMENT_CALIB_SIGMA   = float(os.getenv("SEGMENT_CALIB_SIGMA", "5.0"))
# Pi camera default AE/AWB hunting shifts whole-frame brightness more than
# DroidCam's ISP does, so it needs a wider margin above its noise floor.
SEGMENT_CALIB_SIGMA_PI = float(os.getenv("SEGMENT_CALIB_SIGMA_PI", "9.0"))
# Frames received in the first SEGMENT_CALIB_WARMUP_S seconds after a camera's
# very first frame keep the frame-diff reference fresh but are NOT accumulated
# into the noise sample. Both cameras' auto-exposure/white-balance loops are
# still converging right after stream start, so calibrating immediately can
# measure an artificially quiet moment (2026-07-09: both cameras calibrated in
# under 2s with std_diff of ~0.01-0.02 — so tiny that sigma*std_diff barely
# moved the threshold off the global floor — then a genuine post-settling
# brightness step opened a real-looking-but-empty ~0.5s segment on both
# cameras a few seconds later, twice, ~4s apart). Delaying the start of
# accumulation gives the sensor time to get well into its startup convergence
# so calibration measures steady-state noise instead. Raise this further if
# false-positive segments keep appearing shortly after "calibrated" log lines.
SEGMENT_CALIB_WARMUP_S = float(os.getenv("SEGMENT_CALIB_WARMUP_S", "3.0"))
# Log (DEBUG, mcp_robot.recorder logger) every own-motion trip and the
# resulting shared-clock update — mean_diff/pixel counts vs. thresholds, and
# which camera drove the update. Root logger stays at INFO (server.py), so
# this only bumps the recorder logger specifically; it won't flood the rest
# of the log. Was default-on to diagnose which camera drove long/stale
# segments (see 2026-07-08 investigation); recorder.py now closes all
# cameras' segments off one shared last-motion clock instead of a per-camera
# cross-trigger timestamp, which resolved the race, so this defaults off again.
SEGMENT_MOTION_LOG = bool(int(os.getenv("SEGMENT_MOTION_LOG", "0")))

# ── Thought/planning segments ────────────────────────────────────────────────
# SegmentRecorder.log_thought() synthesizes a frozen-frame segment (no real
# motion) so a non-motion diagnostic moment still appears as a subtitled
# scene in the merged video.
THOUGHT_SEGMENT_DURATION_S = float(os.getenv("THOUGHT_SEGMENT_DURATION_S", "3.0"))

# ── Merged (tiled) task video ─────────────────────────────────────────────────
# compile_video.py --camera merged (or compile_video(camera="merged")) tiles
# the two cameras plus a subtitle card into one video:
#   left-top = subtitle (observation, then action) on black
#   left-bottom = front (Pi) camera
#   right = external (DroidCam) camera, full column height
# MERGE_CELL_HEIGHT is the height of each left-column pane; cell width is
# derived from the Pi camera's native aspect ratio. The right pane is scaled
# to double that height (to fill the column) at the DroidCam's native aspect.
MERGE_CELL_HEIGHT = int(os.getenv("MERGE_CELL_HEIGHT", "360"))
# Matches SEGMENT_FPS_DROIDCAM/SEGMENT_FPS_PI (both real, remuxed capture rates, not
# just declared labels) so the fps= filter in _scale_pad is a pass-through rather than
# a frame-duplicating upsample.
MERGE_FPS = float(os.getenv("MERGE_FPS", "30.0"))
# Segments from either camera within this many seconds of each other are
# treated as one "scene" (tiled together into one clip) rather than being
# sequential scenes. Derived from the recorder's own pre-roll/cooldown knobs
# so both cameras' segments for the same action reunite even though the
# shared-motion-clock in recorder.py only guarantees they start/end within
# about one frame interval of each other, not exactly.
MERGE_SCENE_GAP_S = float(os.getenv(
    "MERGE_SCENE_GAP_S",
    str(SEGMENT_PREROLL_S + SEGMENT_COOLDOWN_S),
))

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

# Model used for free-text object localization (locate_object tool / VLM fallback).
# Flash is fast and cheap; override with LOCATE_OBJECT_MODEL if needed.
# LOCATE_OBJECT_FALLBACK_MODEL is tried on quota exhaustion (429/RESOURCE_EXHAUSTED).
# It has its own independent per-model RPD quota, so the pair stacks rather than
# sharing one daily budget.
LOCATE_OBJECT_MODEL          = os.getenv("LOCATE_OBJECT_MODEL",          "gemini-3.5-flash")
LOCATE_OBJECT_FALLBACK_MODEL = os.getenv("LOCATE_OBJECT_FALLBACK_MODEL", "gemini-2.5-flash")

# Skip VQA verification for lower_arm (the action is reliable enough).
# Set LOWER_ARM_VQA=1 to re-enable.
LOWER_ARM_VQA = bool(os.getenv("LOWER_ARM_VQA", ""))

# exp:49e9ab56 — lift_arm is safe/reliable enough to skip VQA, same as
# control_gripper's "open" action. Set LIFT_ARM_VQA=1 to re-enable.
LIFT_ARM_VQA = bool(os.getenv("LIFT_ARM_VQA", ""))

# click_button's VQA verdicts on the press/release have proven unreliable,
# while the action itself (measured-distance press + fractional release) is
# reliable enough for the current experiment. Skip VQA for now. Set
# CLICK_BUTTON_VQA=1 to re-enable.
CLICK_BUTTON_VQA = bool(os.getenv("CLICK_BUTTON_VQA", ""))

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
