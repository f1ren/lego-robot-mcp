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

## pddl_consult_test.py — replay consult_vqa_for_pddl_domain

Simulates the `consult_vqa_for_pddl_domain` MCP tool against saved images instead of a
live robot. `QUESTION_TEXT` is imported from `mcp_robot.vision.CONSULT_DOMAIN_QUESTION` —
the same constant the production tool sends — so editing it there and re-running this
script is enough to try new wording; there is nothing to keep in sync separately.
Defaults to the images/context from a real failed call (see `tests/fixtures/pddl_consult/`).

```bash
# Re-run the default (real) failure scenario
python3 tests/vqa/pddl_consult_test.py

# Try a different model
GEMINI_MODEL=gemini-2.5-pro python3 tests/vqa/pddl_consult_test.py

# Point at a different scenario / domain
python3 tests/vqa/pddl_consult_test.py \
    --front /path/to/pi_camera.jpg --external /path/to/droidcam.jpg \
    --context "The gripper closed on empty air; the cup was 5cm left of center." \
    --domain pddl/robot_domain.pddl.bak

# Run the same prompt/images N times before trusting a wording — Gemini is
# stochastic, so one good response doesn't mean the wording is reliable.
# --expect is a local substring tally only (never sent to the model); keep
# scenario-specific hints out of QUESTION_TEXT itself so it stays reusable
# across unrelated experiments.
python3 tests/vqa/pddl_consult_test.py --repeat 5 --expect SUBSTRING1 --expect SUBSTRING2
```

Because both consumers import the same `CONSULT_DOMAIN_QUESTION` constant, there is no
separate "port it back" step — editing `mcp_robot/vision.py` updates the live
`consult_vqa_for_pddl_domain` tool immediately (the MCP server hot-reloads on code
changes).
