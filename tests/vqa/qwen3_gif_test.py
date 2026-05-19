#!/usr/bin/env python3
"""
Send PROMPT_TEXT + robot-rotating.gif to local Qwen3-VL via Ollama.

GIF frames are extracted and passed as sequential images (Ollama images= field).

Usage:
  python qwen3_gif_test.py
  OLLAMA_HOST=http://localhost:11434 OLLAMA_MODEL=qwen3-vl python qwen3_gif_test.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:32b-thinking")
GIF_PATH     = Path(__file__).parent / "robot-rotating.gif"

PROMPT_TEXT = (
    """You are analysing a 4-motor Lego robot (left wheel, right wheel, arm, gripper). The gripper defines the robot's front.
ACTION COMMANDED: drive left=60 right=60 for 0.8s EXPECTED OUTCOME: Moving the
gripper directly over the paper ball to pick it up. Ball is ~15cm ahead.
Previous attempt at speed 40 for 0.5s barely moved (floor slippage). Goal:
gripper positioned just around the ball, ready to lower and close. Attached is a
video of camera captured during the action.
IMPORTANT: Describe turning direction as clockwise/counter-clockwise (viewed from above) or compass directions (north/south/east/west), not as 'left' or 'right'.
Consider the robot orientation. Did it rotate?
Reply in EXACTLY this format on two
lines: Verdict: YES | NO | PARTIAL —  Changes: <1-2 short sentences on what
actually happened during the motion>"""
)

# PROMPT_TEXT = "Describe the motion in this video"

MAX_FRAMES = 16  # cap to avoid overwhelming the context window


def extract_gif_frames(gif_path: Path) -> list[bytes]:
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow not installed — run: pip install Pillow")

    frames: list[bytes] = []
    with Image.open(gif_path) as im:
        try:
            while True:
                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="JPEG", quality=85)
                frames.append(buf.getvalue())
                im.seek(im.tell() + 1)
        except EOFError:
            pass

    if len(frames) > MAX_FRAMES:
        step = len(frames) / MAX_FRAMES
        frames = [frames[int(i * step)] for i in range(MAX_FRAMES)]

    return frames


def main() -> None:
    if not GIF_PATH.exists():
        sys.exit(f"GIF not found: {GIF_PATH}")

    try:
        import ollama
    except ImportError:
        sys.exit("ollama package not installed — run: pip install ollama")

    frames = extract_gif_frames(GIF_PATH)

    print("=" * 70)
    print(f"MODEL  : {OLLAMA_MODEL}")
    print(f"HOST   : {OLLAMA_HOST}")
    print(f"GIF    : {GIF_PATH}  ({GIF_PATH.stat().st_size:,} bytes)")
    print(f"FRAMES : {len(frames)}")
    print()
    print("--- PROMPT ---")
    print(PROMPT_TEXT)
    print("--- END PROMPT ---")
    print("=" * 70)
    print()

    print("Calling Ollama... ", end="", flush=True)
    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": PROMPT_TEXT, "images": frames}],
    )
    print("done\n")

    print("--- RESPONSE ---")
    print(resp["message"]["content"].strip())
    print("--- END RESPONSE ---")


if __name__ == "__main__":
    main()
