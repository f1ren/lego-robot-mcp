# Lego Robot MCP

An [MCP](https://modelcontextprotocol.io) server that lets Claude drive a 4-motor LEGO robot.
It controls a Raspberry Pi with a [Build HAT](https://www.raspberrypi.com/products/build-hat/),
sees through a Pi Camera on the robot, and uses an iPhone that overlooks the scene as an external camera.

## Hardware

- Raspberry Pi 4 + Build HAT, with the HAT's own 8 V / 48 W barrel-jack power supply (motors don't run on USB power)
- 4 LEGO motors: **A** left wheel, **B** right wheel, **C** gripper, **D** arm
- Pi Camera v1 (OV5647) on the robot, facing forward
- An iPhone running SimpleIPCamera, overlooking the robot

## Setup

### 1. Raspberry Pi

1. Flash **Raspberry Pi OS Bookworm** with [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
   In the customisation step, set hostname `rpi`, username `rpi`, your Wi-Fi, and enable SSH
   ([guide](https://www.raspberrypi.com/documentation/computers/getting-started.html)).
2. Set up key-based SSH from your computer, because the server logs in without a password
   ([SSH docs](https://www.raspberrypi.com/documentation/computers/remote-access.html)):
   ```bash
   ssh-copy-id rpi@rpi.local
   ssh rpi@rpi.local          # should log in without asking for a password
   ```
3. Enable the serial port that the Build HAT talks over
   ([Build HAT docs](https://www.raspberrypi.com/documentation/accessories/build-hat.html)):
   run `sudo raspi-config` → Interface Options → Serial Port → login shell **No**, serial hardware **Yes**, then reboot.
4. Install the Python libraries on the Pi. The server runs its scripts over SSH with the system `python3`:
   ```bash
   sudo apt install python3-build-hat python3-picamera2 gpiod lsof   # gpiod/lsof: to reset a stuck HAT
   ```
5. Check the camera and a motor
   ([camera docs](https://www.raspberrypi.com/documentation/computers/camera_software.html),
   [buildhat library](https://buildhat.readthedocs.io/)):
   ```bash
   rpicam-hello --list-cameras
   python3 -c "from buildhat import Motor; print(Motor('A').get_position())"
   ```

If you use a different hostname or user, set `ROBOT_HOST` / `ROBOT_USER` in `.mcp.json`.
If your motors are wired differently, set `PORT_LEFT_WHEEL`, `PORT_RIGHT_WHEEL`, `PORT_GRIPPER` and `PORT_ARM`.

### 2. SimpleIPCamera (iPhone)

1. Install [SimpleIPCamera](https://apps.apple.com/app/id6752223439) and join the same Wi-Fi as the Pi.
2. Give the phone the fixed IP `192.168.8.190`. Use a DHCP reservation on your router, or set it on the phone:
   Settings → Wi-Fi → ⓘ → Configure IP → Manual.
   You can also leave the phone's IP as it is and set `SIMPLEIPCAMERA_URL` in `.mcp.json` to its address instead.
3. In the app, set the resolution to **720p** and tap **Start Server**.
   The server reads `http://192.168.8.190:8080/stream.mjpeg`.
4. Keep the app in the foreground, because iOS stops the camera when the app goes to the background or the phone locks:
   - Settings → Display & Brightness → Auto-Lock → **Never**
   - Low Power Mode off
   - Phone on a charger
5. The app serves one viewer at a time. If you check the stream in a browser, close the tab afterwards.

The server rotates the frames 90° clockwise (`SIMPLEIPCAMERA_ROTATION`). Change that value if the image comes out sideways.

### 3. This repo

```bash
git clone https://github.com/f1ren/lego-robot-mcp && cd lego-robot-mcp
python3 -m venv .venv                                   # Python 3.11+
.venv/bin/pip install -e ".[viz]" mcpmon mcp-memory-service
export GEMINI_API_KEY=...                               # .mcp.json reads it from your environment
```

PDDL planning runs on a separate MCP server, [NAPC](https://github.com/f1ren/NAPC). Install it as its README describes.
`.mcp.json` contains absolute paths from the author's machine (the `napc` command, the memory DB, and the NAPC output paths). Change them to yours.

## Scene text

While a task runs, the server reads new text in both camera streams, such as a sign saying which bin takes paper. It only reads a view that is still, clear, and changed since the last read. It passes on only text containing a word it hasn't seen before. That text goes to NAPC, which checks it against the current plan. If the text changes what the plan should do, NAPC recommends halting, adapts the PDDL domain/problem, and replans. Its advisories show up as `napc_advisory` in every tool result of this server.

The two servers exchange findings and advisories through one directory: `SCENE_EVENTS_DIR` here and `NAPC_EVENTS_DIR` for napc, both `output/napc_events` in `.mcp.json`. The frames behind each finding, with an `_ocr.jpg` overlay of what was read, are saved in `output/scene_text/`. The settings are the `SCENE_TEXT_*` entries in `mcp_robot/config.py`. See [mcp_robot/scene_text.py](mcp_robot/scene_text.py) for the design.

## Usage

Start Claude Code in the repo with `claude`. It starts the servers listed in `.mcp.json`, and `/mcp` shows their status.
Then give it a task, e.g. "grab the red cup", or run `/grab`.

- Logs: `output/logs/mcp_server.log`
- All settings: `mcp_robot/config.py`. Each one can be overridden with an environment variable in `.mcp.json`.
- The rules Claude follows in this project: [CLAUDE.md](CLAUDE.md)
