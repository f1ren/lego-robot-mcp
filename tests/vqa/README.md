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
python3 tests/qwen_test.py --mode clip --latest --camera simpleipcamera
```

Override model/host via env vars:

```bash
OLLAMA_HOST=http://rpi.local:11434 OLLAMA_MODEL=qwen2.5vl python3 tests/qwen_test.py --latest
```

## Replaying consult_vqa_for_pddl_domain

The PDDL-repair-via-VLM replay script (formerly `pddl_consult_test.py` here) now lives in
the NAPC repo (`~/Projects/NAPC`, `f1ren/NAPC` on GitHub) as `scripts/replay_consult.py` —
it doesn't touch robot physics, only the prompt/extraction logic that lego-robot-mcp's
`consult_vqa_for_pddl_domain` calls into, so it belongs with that package rather than here.
Same fixture images (`scripts/fixtures/front.jpg` / `external.jpg` there are byte-identical
to the old `tests/fixtures/pddl_consult/` copies), same CLI shape (`--repeat`/`--expect`/
`--save-domain`/`--save-problem`/etc.), run from the NAPC checkout instead:

```bash
cd ~/Projects/NAPC
python3 scripts/replay_consult.py
```

One capability gap versus the old lego-robot-mcp script: `replay_consult.py` only calls
`gemini_ask_with_images` — it has no `--backend ollama` option, even though
`neurosymbolic_counselor.backends.ollama_ask_with_images` exists. Add one if you need to
replay a scenario against the local Ollama/Qwen backend.
