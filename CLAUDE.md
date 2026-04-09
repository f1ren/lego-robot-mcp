# CLAUDE.md — Lego Robot MCP Server

## Project context
This repo implements an MCP server that lets Claude Code (and other MCP clients) control a Lego robot attached to a Raspberry Pi via the BuildHat HAT and OV5647 Pi Camera.

The robot is accessible at `ssh rpi@rpi.local` (no password prompt required — key-based auth).

## Hardware facts (verified 2026-04-09)
- **RPi SSH**: `rpi@rpi.local` — reachable, key-auth configured
- **Camera**: OV5647 (Pi Camera v1), detected on `/dev/media0`, tuning file `ov5647.json`
- **Camera resolution**: 640×480 works well; 1640×1232 max for OV5647
- **Motors**: BuildHat on ports A–D; A and B confirmed motor-type devices
- **BuildHat Python lib**: `from buildhat import Motor, Hat` — both importable

## Key rules

### Never assume motor ports
Always call `get_motor_position` (or `Motor(port).get_position()`) before issuing a move command. Ports that have no motor attached raise a `DeviceError` — catch it and report clearly.

### Always verify after action
After any MOVE/GRASP/PUT:
1. Read motor positions and confirm the delta matches intent.
2. Call `capture_image` and inspect the frame to confirm the visual outcome.
3. If either check fails, stop and report — do not retry blindly.

### SSH execution pattern
All RPi code runs over SSH via `paramiko` (not `subprocess`). Use the `rpi_client.py` helper:
```python
result = rpi.run_python("""
from buildhat import Motor
m = Motor('A')
m.run_for_degrees(90)
print(m.get_position())
""")
```
Never construct shell strings with user-provided values (command injection risk). Use the structured helpers.

### Camera capture
`picamera2` must be started fresh each call (no persistent daemon process — the library serializes access). Expected log noise from `libcamera` is normal and can be suppressed by redirecting stderr on the RPi side.

### MCP tool contract
Every tool must return a JSON-serializable dict with at least:
```json
{
  "ok": true,
  "data": { ... },
  "error": null
}
```
On failure: `"ok": false, "error": "<human-readable message>"`.

## Development workflow

```bash
# Run the MCP server locally (requires SSH access to RPi)
python3 -m mcp_robot.server

# Test a single tool without Claude Code
python3 -m mcp_robot.server --test get_motor_position --port A

# Lint
ruff check mcp_robot/

# Tests (unit tests mock SSH; integration tests need live RPi)
pytest tests/unit/
pytest tests/integration/  # requires RPi on network
```

## Stop conditions — hard stops
If any of these occur **stop immediately and tell the user**:
- A motor port raises `DeviceError` (motor not connected or wrong type)
- Camera capture returns empty bytes or raises an exception
- Motor position after a move does not change within ±5° of expected
- Visual verification shows no change when a physical change was expected

Do not attempt workarounds for hardware faults — they need physical inspection.

## Useful RPi one-liners

```bash
# Check all BuildHat ports
ssh rpi@rpi.local "python3 -c \"from buildhat import Hat; h=Hat(); print(h.port_info())\""

# Quick camera test
ssh rpi@rpi.local "python3 -c \"
from picamera2 import Picamera2; import io, time
c=Picamera2(); c.configure(c.create_still_configuration()); c.start(); time.sleep(1)
b=io.BytesIO(); c.capture_file(b,format='jpeg'); c.stop(); c.close()
print(len(b.getvalue()),'bytes')
\""

# Read all motor positions
ssh rpi@rpi.local "python3 -c \"
from buildhat import Motor
for p in 'ABCD':
    try: print(p, Motor(p).get_position())
    except Exception as e: print(p, 'no device:', e)
\""
```
