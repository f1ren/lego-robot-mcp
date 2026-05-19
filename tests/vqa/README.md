# Tests

## qwen_test.py — standalone Qwen vision tester

Calls Qwen directly with the same prompt and images the MCP server would send.
Prints the full prompt and image paths before the call so you can see exactly what the model receives.

```bash
# Use the latest action_video_* folder (most common)
python3 tests/qwen_test.py --latest \
    --action "drive left=40 right=-40 for 1s" \
    --expected "robot turns clockwise"

# Point at a specific folder
python3 tests/qwen_test.py --mode video \
    --folder /tmp/lego-robot-snapshots/action_video_20260504_134200 \
    --action "drive left=40 right=-40 for 1s" \
    --expected "robot turns clockwise"

# Before/after stills (fallback path)
python3 tests/qwen_test.py --mode stills \
    --before /tmp/lego-robot-snapshots/before_pi_camera.jpg \
    --after  /tmp/lego-robot-snapshots/after_pi_camera.jpg \
    --action "drive left=40 right=-40 for 1s" \
    --expected "robot turns clockwise"

# Clip VQA (used by capture_front/external_video_clip)
python3 tests/qwen_test.py --mode clip --latest --camera droidcam
```

Override model/host via env vars:

```bash
OLLAMA_HOST=http://rpi.local:11434 OLLAMA_MODEL=qwen2.5vl python3 tests/qwen_test.py --latest
```
