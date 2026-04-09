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

## Phase 2 — MCP Server Implementation

### File layout
```
mcp_robot/
├── server.py          # MCP server entry point
├── rpi_client.py      # SSH session manager + RPi command execution
├── robot_actions.py   # High-level GRASP, PUT, MOVE logic
├── vision.py          # Computer vision helpers (position check, object detect)
├── requirements.txt
└── pyproject.toml     # MCP tool declarations
```

### MCP Tools to expose

| Tool name | Description |
|---|---|
| `get_motor_position` | Read current absolute position (degrees) of a motor port (A/B/C/D) |
| `move_motor` | Move a motor by N degrees at given speed; returns final position |
| `run_motors` | Move multiple motors simultaneously |
| `capture_image` | Capture a still frame; returns base64 JPEG + timestamp |
| `move` | High-level: drive the robot in a direction by distance/time |
| `grasp` | High-level: close gripper motor; verify grip with force or position delta |
| `put` | High-level: open gripper motor to release object |
| `look` | Capture image and return it for AI visual inspection |
| `get_robot_state` | Return all motor positions + a camera snapshot in one call |

### Motor port mapping (to be confirmed with physical robot)
| Port | Motor | Role |
|---|---|---|
| A | Large Lego motor | Left drive / arm extension |
| B | Large Lego motor | Right drive / arm extension |
| C | Medium motor | Gripper / wrist |
| D | Medium motor | Gripper / wrist (if dual) |

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
