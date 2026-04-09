# MCP Server — Lego Robot on Raspberry Pi with BuildHat + Camera

## Project Goal
Expose a Lego robot (controlled via BuildHat motors and a Pi Camera) as an MCP server so AI tools like Claude Code can plan and execute robot actions: MOVE, GRASP, PUT, and others.

## Architecture

```
Claude Code (Mac)
    │
    ▼
MCP Server (runs locally on Mac)
    │  SSH (rpi@rpi.local)
    ▼
Raspberry Pi
    ├── BuildHat  →  Lego Motors (ports A, B, C, D)
    └── Pi Camera (OV5647) via picamera2
```

The local MCP server maintains a persistent SSH connection (via `paramiko`) to the RPi and:
- Executes BuildHat commands to read/write motor state
- Captures camera frames and returns them as base64 JPEG
- Implements high-level robot actions (GRASP, PUT, MOVE) built on top of the primitives
- Uses computer vision (OpenCV or PIL) on captured frames to verify actions

---

## Phase 1 — RPi Environment Setup (prerequisite)

Verified working (2026-04-09):
- [x] `ssh rpi@rpi.local` reachable
- [x] `buildhat` Python package importable
- [x] `picamera2` importable, OV5647 camera detected at `/base/soc/i2c0mux/i2c@1/ov5647@36`
- [x] Motor A: readable position (port A)
- [x] Motor B: readable position (port B)
- [x] Camera capture: 640×480 JPEG, ~56 KB

### If setup is needed on a fresh RPi
```bash
# Enable camera interface
sudo raspi-config nonint do_camera 0

# Install BuildHat library
pip3 install buildhat

# Install picamera2 (usually pre-installed on Raspberry Pi OS)
sudo apt-get install -y python3-picamera2

# Install CV dependencies for action verification
pip3 install opencv-python-headless numpy
```

---

## Phase 2 — MCP Server Implementation ✓ Done (2026-04-09)

### File layout
```
mcp_robot/
├── __init__.py
├── config.py      # env-based configuration
├── rpi_client.py  # persistent SSH session, Python execution on RPi
├── robot.py       # motor primitives + high-level GRASP/PUT/MOVE
├── camera.py      # still + video clip capture via picamera2
├── vision.py      # Gemini vision analysis (google-genai SDK)
└── server.py      # FastMCP entry point — 13 tools
requirements.txt
pyproject.toml
.mcp.json
.venv/             # Python 3.12 venv (mcp, paramiko, google-genai, Pillow)
```

### MCP Tools (all verified importable)

| Tool | Description |
|---|---|
| `get_robot_state` | Motor positions + live camera frame — call before planning |
| `get_motor_positions` | Positions for all 4 ports |
| `move_motor` | Move a single port by N degrees |
| `drive` | forward / backward / left / right / stop |
| `move_arm` | Move arm up or down |
| `control_gripper` | Open or close gripper |
| `grasp` | Lower arm + close gripper |
| `put` | Open gripper + raise arm |
| `capture_image` | Still frame as MCP `ImageContent` |
| `capture_video_clip` | N-second clip as sequence of `ImageContent` frames |
| `analyze_scene` | Capture + Gemini description |
| `analyze_video_clip` | Capture clip + Gemini temporal reasoning |
| `verify_action` | Before/after frames → Gemini success judgement |

### Motor port mapping (default — adjust via env vars if wiring differs)
| Port | Env var | Default | Role |
|---|---|---|---|
| A | `PORT_LEFT_WHEEL` | A | Left drive wheel |
| B | `PORT_RIGHT_WHEEL` | B | Right drive wheel |
| C | `PORT_ARM` | C | Arm up/down |
| D | `PORT_GRIPPER` | D | Gripper open/close |

### Implementation notes
- SSH execution uses `os.dup2(devnull, 2)` to silence libcamera C-level logs
- Exceptions in RPi scripts are caught and returned as `{"__error__": ...}` JSON
- Video clip capture uses `format='RGB888'` to avoid PIL/XBGR array type mismatch
- Vision uses `google-genai` SDK (v1+); `google-generativeai` is deprecated

---

## Phase 3 — Verification Protocol

Each action tool must be verifiable. Verification approach:

### Motor verification
1. Read position **before** action (`get_motor_position`)
2. Execute action (`move_motor`)
3. Read position **after** action
4. Assert `|final - expected| < 5°`

### Camera / CV verification
1. Capture image before action
2. Execute action
3. Capture image after action
4. Use OpenCV to detect positional change (contour diff, color blob shift, ArUco marker)
5. Report whether visual change matches expected outcome

### End-to-end GRASP/PUT verification
- Before GRASP: detect object in frame, gripper open
- After GRASP: detect gripper closed (motor delta), object position changed or absent in frame
- After PUT: object visible at target location, gripper open

---

## Phase 4 — Claude Code Integration

Add `.mcp.json` to the project root so Claude Code auto-discovers the server:

```json
{
  "mcpServers": {
    "lego-robot": {
      "command": "python3",
      "args": ["-m", "mcp_robot.server"],
      "env": {
        "ROBOT_HOST": "rpi.local",
        "ROBOT_USER": "rpi"
      }
    }
  }
}
```

Once running, Claude Code can call tools like:
```
use_mcp_tool lego-robot capture_image {}
use_mcp_tool lego-robot grasp {}
use_mcp_tool lego-robot get_motor_position {"port": "A"}
```

---

## Stop Conditions

Stop and report to the user if:
- `buildhat` import fails on RPi → BuildHat HAT may not be seated or firmware needs update
- Any motor port returns an error → wrong port, motor not connected, or motor type mismatch
- `picamera2` import fails or camera not found → camera cable not connected or interface disabled
- Camera capture returns 0 bytes → camera hardware fault

All of these were passing as of initial verification (2026-04-09).
